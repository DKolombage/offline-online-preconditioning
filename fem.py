import numpy as np
import scipy.sparse as sparse
import util

def localStiffnessMatrix_hetero(world, a_local, KInd):
    """
    Assemble the local stiffness matrix on one coarse element with
    heterogeneous fine-scale coefficient a_local.

    Parameters
    ----------
    world : World
        Global mesh object (contains NWorldFine, NCoarseElement, etc.).
    a_local : ndarray
        Flattened fine-scale coefficient values for this element.
        Typically shape = (Ne[0] * Ne[1],)
    KInd : tuple(int,int)-- Coarse element index (ex, ey).

    Returns
    -------
    K_T : csr_matrix, shape (nloc, nloc)
        Local fine-scale stiffness matrix on this element.
    """
    Ne = np.asarray(world.NCoarseElement, dtype=int)
    d  = len(Ne)
    if d != 2:
        raise NotImplementedError("localStiffnessMatrix_hetero currently supports 2D only.")

    nx, ny = Ne
    hx, hy = 1.0 / (world.NWorldFine[0]), 1.0 / (world.NWorldFine[1])

    nloc = (nx + 1) * (ny + 1)
    K_T = sparse.lil_matrix((nloc, nloc))

    # Loop over each fine cell in this coarse element
    for iy in range(ny):
        for ix in range(nx):
            elem_id = iy * nx + ix
            a_val = float(a_local[elem_id])

            # Local node indices (4 Q1 nodes per fine cell)
            n00 = iy * (nx + 1) + ix
            n10 = n00 + 1
            n01 = n00 + (nx + 1)
            n11 = n01 + 1
            nodes = [n00, n10, n01, n11]

            # Local stiffness for one fine cell (Q1 element)
            K_ref = (a_val / (hx * hy)) * np.array([
                [ hy/hx + hx/hy, -hy/hx,        -hx/hy,         0.0        ],
                [-hy/hx,          hy/hx + hx/hy, 0.0,          -hx/hy      ],
                [-hx/hy,          0.0,           hy/hx + hx/hy, -hy/hx      ],
                [0.0,             -hx/hy,        -hy/hx,        hy/hx + hx/hy]
            ]) * 0.25  # standard bilinear scaling

            for i_loc, i_gl in enumerate(nodes):
                for j_loc, j_gl in enumerate(nodes):
                    K_T[i_gl, j_gl] += K_ref[i_loc, j_loc]

    return K_T.tocsr()

def localProlongationMatrix(NCoarseElement):
    """
    Build the local prolongation matrix P_T for one coarse element.

    Parameters
    ----------
    NCoarseElement : array-like of length 2
        Number of fine elements per coarse element, e.g. [8,8].

    Returns
    -------
    P_T : csr_matrix, shape (n_fine_local, 4)
        Local prolongation operator (fine <- coarse) on one coarse element.
    """
    NEx, NEy = NCoarseElement
    nx, ny = NEx + 1, NEy + 1
    Nloc = nx * ny  # local fine nodes

    # Local coarse vertex coordinates (bilinear Q1 shape functions)
    x_c = np.array([0.0, 1.0])
    y_c = np.array([0.0, 1.0])

    # Fine-grid coordinates (normalized to [0,1])
    x_f = np.linspace(0, 1, nx)
    y_f = np.linspace(0, 1, ny)

    # Bilinear Q1 basis functions on coarse element
    def phi1(x, y): return (1 - x) * (1 - y)
    def phi2(x, y): return x * (1 - y)
    def phi3(x, y): return (1 - x) * y
    def phi4(x, y): return x * y

    # Construct interpolation matrix
    rows, cols, vals = [], [], []
    for iy, y in enumerate(y_f):
        for ix, x in enumerate(x_f):
            row = iy * nx + ix
            vals_loc = [phi1(x, y), phi2(x, y), phi3(x, y), phi4(x, y)]
            for j, v in enumerate(vals_loc):
                rows.append(row)
                cols.append(j)
                vals.append(v)

    P_T = sparse.csr_matrix((vals, (rows, cols)), shape=(Nloc, 4))
    return P_T

#----------
def assemblePatchMassMatrix(NPatchFine):
    """
    Assemble the patch-wide mass matrix from local mass matrices.

    Parameters
    ----------
    NPatchFine : array-like
        Number of fine elements in each direction of the patch.

    Returns
    -------
    MPatch : sparse matrix
        Assembled patch-local mass matrix.
    """
    ALoc = localMassMatrix(NPatchFine)
    MPatch = assemblePatchMatrix(NPatchFine, ALoc)
    return MPatch

## ------
def localMatrix(d, matrixValuesTakesBinaryIndices):
    def convertToBinaryIndices(f):
        return lambda *ind: f(np.array(ind[:d], dtype='bool'),
                              np.array(ind[d:], dtype='bool'))

    ABin = np.fromfunction(convertToBinaryIndices(matrixValuesTakesBinaryIndices), shape=[2]*(2*d), dtype='int64')
    AFlat = ABin.flatten()
    A = AFlat.reshape(2**d, 2**d)
    return A
    

def localMassMatrix(N):
    d = np.size(N)
    detJ = np.prod(1./N)
    def massMatrixBinaryIndices(ib, jb):
        return detJ*(1 << np.sum(~(ib ^ jb), axis=0))/6.**d
    
    return localMatrix(d, massMatrixBinaryIndices)
    
def localStiffnessMatrix(N):
    d = np.size(N)
    detJ = np.prod(1./N)
    def stiffnessMatrixBinaryIndices(ib, jb):
        M = detJ*(1 << np.sum(~(ib ^ jb), axis=0))/6.**d
        A = M*np.sum(list(map(np.multiply, N**2, 3*(1-3*(ib ^ jb)))), axis=0)
        return A
    
    return localMatrix(d, stiffnessMatrixBinaryIndices)

def localBoundaryNormalDerivativeMatrixGetter(N):
    return lambda k, neg: localBoundaryNormalDerivativeMatrix(N, k, neg)

def localBoundaryNormalDerivativeMatrix(N, k=0, neg=False):
    d = np.size(N)
    notk = np.ones_like(N,dtype='bool')
    notk[k] = False
    detJk = np.prod(1./N[notk])
    def boundaryNormalDerivativeMatrixBinaryIndices(ib, jb):
        C = detJk*(1 << np.sum(~(ib[notk] ^ jb[notk]), axis=0))/6.**(d-1)
        C *= N[k]*(1-2*(jb[k]^neg))*(1-(ib[k]^neg))
        return C
    
    return localMatrix(d, boundaryNormalDerivativeMatrixBinaryIndices)

def localBoundaryMassMatrixGetter(N):
    return lambda k, neg: localBoundaryMassMatrix(N, k, neg)

def localBoundaryMassMatrix(N, k=0, neg=False):
    d = np.size(N)
    notk = np.ones_like(N,dtype='bool')
    notk[k] = False
    detJk = np.prod(1./N[notk])
    def boundaryMassMatrixBinaryIndices(ib, jb):
        C = detJk*(1 << np.sum(~(ib[notk] ^ jb[notk]), axis=0))/6.**(d-1)
        C *= (1-(ib[k]^neg))*(1-(jb[k]^neg))
        return C
    
    return localMatrix(d, boundaryMassMatrixBinaryIndices)

def localToPatchSparsityPattern(NPatch, NSubPatch=None):
    if NSubPatch is None:
        NSubPatch = NPatch
        
    d = np.size(NPatch)
    
    loc2Patch = util.lowerLeftpIndexMap(np.ones(d, 'int64'), NPatch)
    pInd = util.lowerLeftpIndexMap(NSubPatch-1, NPatch)

    loc2PatchRep = np.repeat(loc2Patch, 2**d)
    loc2PatchTile = np.tile(loc2Patch, 2**d)
    indexMatrixRows = np.add.outer(pInd, loc2PatchRep)
    indexMatrixCols = np.add.outer(pInd, loc2PatchTile)

    rows = indexMatrixRows.flatten()
    cols = indexMatrixCols.flatten()

    return rows, cols

def assemblePatchMatrix(NPatch, ALoc, aPatch=None):
    d = np.size(NPatch)
    Np = np.prod(NPatch+1)
    Nt = np.prod(NPatch)
    
    if aPatch is None:
        aPatch = np.ones(Nt)

    rows, cols = localToPatchSparsityPattern(NPatch)

    assert((aPatch.ndim == 1 and ALoc.ndim == 2) or
           (aPatch.ndim == 3 and ALoc.ndim == 4))
    
    if aPatch.ndim == 1:
        # Coefficient is scalar
        values = np.kron(aPatch, ALoc.flatten())
    elif aPatch.ndim == 3:
        # Coefficient is matrix-valued
        values = np.einsum('ijkl,Tkl->Tij', ALoc, aPatch).flatten()
    #print(f"len(values): {len(values)}, len(rows): {len(rows)}, len(cols): {len(cols)}")
    #print(f"aPatch.shape: {aPatch.shape}, NPatchFine: {NPatch}, ALoc shape: {ALoc.shape}")

    APatch = sparse.csc_matrix((values, (rows, cols)), shape=(Np, Np))
    APatch.eliminate_zeros()
    
    return APatch

def assemblePatchBoundaryMatrix(NPatch, CLocGetter, aPatch=None, boundaryMap=None):
    # Integral over part of boundary can be implemented by adding an
    # input "chi" as an indicator function to be callable with edge
    # midpoints as inputs.
    d = np.size(NPatch)
    Np = np.prod(NPatch+1)
    Nt = np.prod(NPatch)

    if aPatch is None:
        aPatch = np.ones(Nt)

    if boundaryMap is None:
        boundaryMap = np.ones([d,2], dtype='bool')

    rows = []
    cols = []
    values = []
    # Loop through each dimension
    for k in range(d):
        NEdge = NPatch.copy()
        NEdge[k] = 1
        
        edgeElementInd0 = util.lowerLeftpIndexMap(NEdge-1, NPatch-1)
        rows0, cols0 = localToPatchSparsityPattern(NPatch, NSubPatch=NEdge)

        for neg in [False, True]:
            if boundaryMap[k][int(neg)]:
                CLoc = CLocGetter(k, neg)
                if not neg:
                    edgeElementIndneg = edgeElementInd0
                    rowsneg = rows0
                    colsneg = cols0
                else:
                    pointIndexDisplacement = int(np.prod(NPatch[:k]+1)*(NPatch[k]-1))
                    elementIndexDisplacement = int(np.prod(NPatch[:k])*(NPatch[k]-1))
                    edgeElementIndneg = edgeElementInd0 + elementIndexDisplacement
                    rowsneg = rows0 + pointIndexDisplacement
                    colsneg = cols0 + pointIndexDisplacement

                valuesneg = np.kron(aPatch[edgeElementIndneg], CLoc.flatten())

                rows = np.hstack([rows, rowsneg])
                cols = np.hstack([cols, colsneg])
                values = np.hstack([values, valuesneg])


    APatch = sparse.csc_matrix((values, (rows, cols)), shape=(Np, Np))
    APatch.eliminate_zeros()
    
    return APatch

def localStiffnessTensorMatrixCoefficient(N):
    d = np.size(N)
    detJ = np.prod(1./N)
    ALocTensor = np.zeros([2**d, 2**d, d, d])
    for i in range(2**d):
        ib = np.unpackbits(np.array(i, dtype='uint8'))[:-d-1:-1].astype('bool')
        for j in range(2**d):
            jb = np.unpackbits(np.array(j, dtype='uint8'))[:-d-1:-1].astype('bool')
            for k in range(d):
                for l in range(d):
                    noklMask = np.ones(d, dtype='bool')
                    noklMask[k] = False
                    noklMask[l] = False
                    
                    NonDifferentiatedFactorsNokl = float(1<<np.sum(~(ib[noklMask]^jb[noklMask]))) /(6.**(np.sum(noklMask)))
                    NonDifferentiatedFactorskl = 1.0 if k == l else 1./4
                    DifferentiatedFactors = (1-2*(ib[k]^jb[l]))
                    ALocTensor[i, j, k, l] = detJ*N[k]*N[l]*DifferentiatedFactors*NonDifferentiatedFactorskl*NonDifferentiatedFactorsNokl
    return ALocTensor

def localBasis(N):
    d = np.size(N)

    Phis = [1]
    for k in range(d): 
        x = np.linspace(0,1,N[k]+1)
        newPhis0 = []
        newPhis1 = []
        for Phi in Phis:
            newPhis0.append(np.kron(1-x, Phi))
            newPhis1.append(np.kron(x, Phi))
        Phis = newPhis0 + newPhis1
    return np.column_stack(Phis)

def assembleProlongationMatrix(NPatchCoarse, NCoarseElement): #, localBasis):
    d = np.size(NPatchCoarse)
    Phi = localBasis(NCoarseElement)
    assert np.size(Phi, 1) == 2**d

    NPatchFine = NPatchCoarse*NCoarseElement
    NtCoarse = np.prod(NPatchCoarse)
    NpCoarse = np.prod(NPatchCoarse+1)
    NpFine = np.prod(NPatchFine+1)

    rowsBasis = util.lowerLeftpIndexMap(NCoarseElement, NPatchFine)
    colsBasis = np.zeros_like(rowsBasis)
    
    rowsElement = np.tile(rowsBasis, 2**d)
    colsElement = np.add.outer(util.lowerLeftpIndexMap(np.ones(d, dtype='int64'), NPatchCoarse), colsBasis).flatten()

    rowsOffset = util.pIndexMap(NPatchCoarse-1, NPatchFine, NCoarseElement)
    colsOffset = util.lowerLeftpIndexMap(NPatchCoarse-1, NPatchCoarse)

    rows = np.add.outer(rowsOffset, rowsElement).flatten()
    cols = np.add.outer(colsOffset, colsElement).flatten()
    values = np.tile(Phi.flatten('F'), NtCoarse)

    rows, cols, values = util.ignoreDuplicates(rows, cols, values)
    PPatch = sparse.csc_matrix((values, (rows, cols)), shape=(NpFine, NpCoarse))
    PPatch.eliminate_zeros()
    
    return PPatch

def assembleHierarchicalBasisMatrix(NPatchCoarse, NCoarseElement):
    NPatchFine = NPatchCoarse*NCoarseElement
    NpFine = np.prod(NPatchFine+1)

    # Use simplest possible hierarchy, divide by two in all dimensions
    NLevelElement = NCoarseElement.copy()
    while np.all(np.mod(NLevelElement, 2) == 0):
        NLevelElement = NLevelElement // 2

    assert np.all(NLevelElement == 1)
        
    rowsList = []
    colsList = []
    valuesList = []

    # Loop over levels
    while np.all(NLevelElement <= NCoarseElement):
        # Compute level basis functions on fine mesh
        NCoarseLevel = NCoarseElement//NLevelElement
        NPatchLevel = NPatchCoarse*NCoarseLevel
        PLevel = assembleProlongationMatrix(NPatchLevel, NLevelElement)
        PLevel = PLevel.tocoo()
        
        # Rows are ok. Columns must be sparsened to fine mesh.
        colsMap = util.pIndexMap(NPatchLevel, NPatchFine, NLevelElement)
        rowsList.append(PLevel.row)
        colsList.append(colsMap[PLevel.col])
        valuesList.append(PLevel.data)
        
        NLevelElement = 2*NLevelElement

    # Concatenate lists (backwards so that we can ignore duplicates)
    rows = np.hstack(rowsList[::-1])
    cols = np.hstack(colsList[::-1])
    values = np.hstack(valuesList[::-1])

    # Ignore duplicates
    rows, cols, values = util.ignoreDuplicates(rows, cols, values)

    # Create sparse matrix
    PHier = sparse.csc_matrix((values, (rows, cols)), shape=(NpFine, NpFine))
    PHier.eliminate_zeros()

    return PHier

def localFaceMassMatrix(N):
    d = np.size(N)
    FLoc = np.zeros(2*d)
    for k in range(d):
        NFace = np.array(N)
        NFace[k] = 1
        faceArea = np.prod(1./NFace)
        FLoc[2*k:2*k+2] = faceArea
    return FLoc

def assembleFaceConnectivityMatrix(NPatch, FLoc, boundaryMap=None):
    Nt = np.prod(NPatch)

    d = np.size(NPatch)

    if boundaryMap is None:
        boundaryMap = np.zeros([d, 2], dtype='bool')
    
    rowsList = []
    colsList = []
    valuesList = []

    tBasis = util.linearpIndexBasis(NPatch-1)
    for k in range(d):
        for boundary in [0, 1]:
            NPatchBase = np.array(NPatch)
            NPatchBase[k] -= 1
            
            TIndInterior = (1-boundary)*tBasis[k] + util.lowerLeftpIndexMap(NPatchBase-1, NPatch-1)
            nT = np.size(TIndInterior)

            FLocRepeated = np.repeat(FLoc[2*k + boundary], nT)
            
            # Add diagonal elements
            rowsList.append(TIndInterior)
            colsList.append(TIndInterior)
            valuesList.append(FLocRepeated)

            # Add off-diagonal elements
            rowsList.append(TIndInterior + (2*boundary - 1)*tBasis[k])
            colsList.append(TIndInterior)
            valuesList.append(-FLocRepeated/2.)
            
            rowsList.append(TIndInterior)
            colsList.append(TIndInterior + (2*boundary - 1)*tBasis[k])
            valuesList.append(-FLocRepeated/2.)

            # Add boundary diagonal elements, if applies
            if boundaryMap[k, boundary]:
                NPatchBottom = np.array(NPatch)
                NPatchBottom[k] = 1

                TIndBoundary = (boundary)*tBasis[k]*(NPatch[k]-1) + \
                               util.lowerLeftpIndexMap(NPatchBottom-1, NPatch-1)
                nT = np.size(TIndBoundary)
                
                FLocRepeatedBoundary = np.repeat(FLoc[2*k + boundary], nT)

                rowsList.append(TIndBoundary)
                colsList.append(TIndBoundary)
                valuesList.append(FLocRepeatedBoundary)
                
    # Concatenate lists
    rows = np.hstack(rowsList)
    cols = np.hstack(colsList)
    values = np.hstack(valuesList)

    # Create sparse matrix
    FC = sparse.csc_matrix((values, (rows, cols)), shape=(Nt, Nt))

    return FC
