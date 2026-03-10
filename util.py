import copy
import itertools
import numpy as np

from functools import reduce

import world as wrld
import numpy as np
import scipy.sparse as sp

def localCoarseDofs_from_KInd(NWorldCoarse, KInd):
    """
    Return the 4 global coarse node indices (for Q1 elements)
    corresponding to the coarse element with index KInd = (i, j).
    Coarse nodes are numbered lexicographically:
        node_id = j*(NWorldCoarse[0]+1) + i
    """
    i, j = int(KInd[0]), int(KInd[1])
    NxC, NyC = int(NWorldCoarse[0]), int(NWorldCoarse[1])

    # global coarse node indices (bottom-left, bottom-right, top-left, top-right)
    n0 = j * (NxC + 1) + i
    n1 = j * (NxC + 1) + (i + 1)
    n2 = (j + 1) * (NxC + 1) + i
    n3 = (j + 1) * (NxC + 1) + (i + 1)

    return np.array([n0, n1, n2, n3], dtype=int)


def getFineElementIndices(NFine, NCoarseElement, iElementCoarse):
    """
    Return fine-grid node indices for coarse element iElementCoarse,
    consistent with row-major (y outer, x inner) global numbering
    and x-fastest local order matching FEM local matrices.
    """
    NFine = np.asarray(NFine, dtype=int)
    Ne = np.asarray(NCoarseElement, dtype=int)
    KInd = np.asarray(iElementCoarse, dtype=int)
    fine_shape = NFine + 1
    d = len(NFine)

    if d == 1:
        offset = KInd[0] * Ne[0]
        return np.arange(offset, offset + Ne[0] + 1).tolist()

    elif d == 2:
        iy, ix = KInd
        Ny, Nx = fine_shape[1], fine_shape[0]
        offset_x = ix * Ne[0]
        offset_y = iy * Ne[1]
        x = np.arange(offset_x, offset_x + Ne[0] + 1)
        y = np.arange(offset_y, offset_y + Ne[1] + 1)

        # Generate indices with ij indexing and F-order flattening
        X, Y = np.meshgrid(x, y, indexing='ij')
        fine_node_indices = (Y * Nx + X).ravel(order='F')
        return fine_node_indices.tolist()

#--------------
def getCoarseElementNodes(TInd, NCoarse):
    """
    Returns the global coarse node indices (in lexicographic order) for the given coarse element.
    Handles both 1D and 2D cases.
    Returns
    -------
    nodes : list of int
        List of flat indices of the coarse-grid nodes at this element's corners.
    """
    d = len(NCoarse)

    if d == 1:
        (i,) = TInd
        return [i, i + 1]

    elif d == 2:
        i, j = TInd
        Nx, Ny = NCoarse
        stride = Nx + 1
        bottom_left = j * stride + i
        bottom_right = bottom_left + 1
        top_left = bottom_left + stride
        top_right = top_left + 1
        return [bottom_left, bottom_right, top_left, top_right]

    else:
        raise NotImplementedError("getCoarseElementNodes only supports 1D and 2D.")


def LocalToGlobalFine_PatchFromNode(patch): 
    """
    Get global fine node indices for a PatchFromNode instance.
    """
    iPatchCoarse = patch.iPatchWorldCoarse
    NPatchCoarse = patch.NPatchCoarse
    NWorldCoarse = patch.world.NWorldCoarse
    NCoarseElement = patch.world.NCoarseElement

    fineNodeIndices, _ = fineIndicesInPatch(NWorldCoarse, NCoarseElement,
                                                iPatchCoarse, NPatchCoarse)
    return fineNodeIndices

# Extract local fine-scale coefficient for a given coarse element
def extract_aFineLocal_for_coarse_element(aPert, world, elem_idx):
    """
    Return the fine-element coefficient vector on a single coarse element.

    Accepts elem_idx as:
      - 1D: int or (int,)
      - 2D: (ix, iy) tuple or array-like of two ints
    """

    d  = len(world.NWorldFine)
    Ne = np.asarray(world.NCoarseElement, dtype=int)

    if d == 1:
        # accept int, (int,), [int], np.array([int])
        if isinstance(elem_idx, (tuple, list, np.ndarray)):
            i = int(elem_idx[0])
        else:
            i = int(elem_idx)
        i_start = i * Ne[0]
        i_end   = (i + 1) * Ne[0]
        return aPert[i_start:i_end]

    elif d == 2:
        ex, ey = (int(elem_idx[0]), int(elem_idx[1])) if isinstance(elem_idx, (tuple, list, np.ndarray)) \
                 else (int(elem_idx), None)  # will error clearly if not a pair
        nx_fine, ny_fine = world.NWorldFine
        nx_elem, ny_elem = Ne
        ix_start = ex * nx_elem
        ix_end   = (ex + 1) * nx_elem
        iy_start = ey * ny_elem
        iy_end   = (ey + 1) * ny_elem
        aPert_2D = aPert.reshape((nx_fine, ny_fine), order='C')
        aElem    = aPert_2D[ix_start:ix_end, iy_start:iy_end]
        return aElem.flatten(order='C')

    else:
        raise NotImplementedError("Only 1D and 2D supported")

def get_all_interior_patch_nodes(world, k):
    """
    Return all coarse node indices whose 2k-neighborhood fits inside the domain.
    Works in any dimension.
    """
    NCoarseNodes = world.NWorldCoarse + 1  # e.g. [9] in 1D, [9, 9] in 2D

    # For each dimension, compute valid index range
    index_ranges = [range(k, N - k) for N in NCoarseNodes]

    # Cartesian product of valid ranges
    coords = list(itertools.product(*index_ranges))

    # Convert multi-index to linear index
    indices = [convertpCoordIndexToLinearIndex(NCoarseNodes, coord) for coord in coords]

    return indices

def create_all_valid_patches(world, k):
    """
    Create patches centered at all coarse nodes.
    Returns:
    - patches: list of PatchFromNode objects
    - valid_nodes: list of corresponding center node indices
    """
    NCoarseNodes = world.NWorldCoarse + 1
    #valid_nodes = get_all_patch_nodes_with_full_support(world.NWorldCoarse, k)#get_valid_patch_center_nodes(world, k)
    valid_nodes = get_all_interior_patch_nodes(world, k)
    patches = []
    for nodeInd in valid_nodes:
        patch = wrld.PatchFromNode(world, k, nodeInd)
        patches.append(patch)

    return patches
#-----------------------


def linearpIndexBasis(N):
    """Compute basis b to convert from d-dimensional indices to linear indices.

    Example for d=3:
    
    b = linearpIndexBasis(NWorld)
    ind = np.dot(b, [1,2,3])

    ind contains the linear index for point (1,2,3).
    """
    cp = np.cumprod(N+1, dtype='int64')
    b = np.hstack([[1], cp[:-1]])
    return b

def convertpLinearIndexToCoordIndex(N, ind):
    ind = np.array(ind)
    d = np.size(N)
    if ind.ndim > 0:
        m = np.size(ind)
        coord = np.zeros([d, m], dtype='int64')
    else:
        coord = np.zeros([d], dtype='int64')
    basis = linearpIndexBasis(N)
    for i in range(d-1,-1,-1):
        coord[i,...] = ind//basis[i]
        ind -= coord[i,...]*basis[i]
    assert(np.all(ind == 0))
    return coord

def convertpCoordIndexToLinearIndex(N, coord):
    basis = linearpIndexBasis(N)
    return np.dot(coord, basis)

def interiorpIndexMap(N):
    """Compute indices (linear order) of all interior points."""
    preIndexMap = lowerLeftpIndexMap(N-2, N)
    indexMap = np.sum(linearpIndexBasis(N))+preIndexMap
    return indexMap

def boundarypIndexMap(N, boundaryMap=None):
    return boundarypIndexMapLarge(N, boundaryMap)

def boundarypIndexMapLarge(N, boundaryMap=None):
    d = np.size(N)
    if boundaryMap is None:
        boundaryMap = np.ones([d,2], dtype='bool')
    b = linearpIndexBasis(N)
    allRanges = [np.arange(Ni+1) for Ni in N]
    allIndices = np.array([], dtype='int64')
    for k in range(d):
        kRange = copy.copy(allRanges)
        kRange[k] = np.array([], dtype='int64')
        if boundaryMap[k][0]:
            kRange[k] = np.append(kRange[k], 0)
        if boundaryMap[k][1]:
            kRange[k] = np.append(kRange[k], N[k])
        twoSides = np.meshgrid(*kRange)
        twoSidesIndices = reduce(np.add, list(map(np.multiply, b, twoSides))).flatten()
        allIndices = np.hstack([allIndices, twoSidesIndices])
    return np.unique(allIndices)

def extractElementFine(NCoarse,
                       NCoarseElement,
                       iElementCoarse,
                       extractElements=True):
    return extractPatchFine(NCoarse, NCoarseElement, iElementCoarse, 0*iElementCoarse+1, extractElements)

def extractPatchFine(NCoarse,
                     NCoarseElement,
                     iPatchCoarse,
                     NPatchCoarse,
                     extractElements=True):
    NFine = NCoarse*NCoarseElement
    if extractElements:
        fineIndexBasis = linearpIndexBasis(NFine-1)
        patchFineIndexStart = np.dot(fineIndexBasis, iPatchCoarse*NCoarseElement)
        patchFineIndexMap = lowerLeftpIndexMap(NPatchCoarse*NCoarseElement-1, NFine-1)
    else:
        fineIndexBasis = linearpIndexBasis(NFine)
        patchFineIndexStart = np.dot(fineIndexBasis, iPatchCoarse*NCoarseElement)
        patchFineIndexMap = lowerLeftpIndexMap(NPatchCoarse*NCoarseElement, NFine)
    return patchFineIndexStart + patchFineIndexMap

def pIndexMap(NFrom, NTo, NStep):
    """ Create a map of the point indices from one grid to another.
    
    NFrom is the grid on which the map is defined
    NTo   is the grid onto which the map ranges
    NStep is the the number of steps in the To-grid for each step in the From-grid

    Example in 1D)

    NFrom = np.array([2])
    NTo   = np.array([10])
    NStep = np.array([3])

                 v        v        v
      toIndex =  0  1  2  3  4  5  6  7  8  9  10
    fromIndex =  0        1        2
        grids    |--+--+--|--+--+--|--+--+--+--+

    | denotes the nodes of NFrom (and NTo)
    + denotes the nodes of NTo 

    pIndexMap(NFrom, Nto, NStep) returns np.array([0, 3, 6]) marked by 'v' above

    (NTo[-1] is always ignored, so in this 1D-example NTo can be set to anything)
    """
    NTopBasis = linearpIndexBasis(NTo)
    NTopBasis = NStep*NTopBasis

    uLinearIndex = lambda *index: np.tensordot(NTopBasis[::-1], index, (0, 0))
    indexMap = np.fromfunction(uLinearIndex, shape=NFrom[::-1]+1, dtype='int64').flatten()
    return indexMap

def elementpIndexMap(NTo):
    return lowerLeftpIndexMap(np.ones_like(NTo), NTo)

def lowerLeftpIndexMap(NFrom, NTo):
    return pIndexMap(NFrom, NTo, np.ones(np.size(NFrom), dtype='int64'))

def fillpIndexMap(NCoarse, NFine):
    assert np.all(np.mod(NFine, NCoarse) == 0)
    NStep = NFine//NCoarse
    return pIndexMap(NCoarse, NFine, NStep)

def cornerIndices(N):
    return fillpIndexMap(np.ones(np.size(N), dtype='int64'), N)

def numNeighboringElements(iPatchCoarse, NPatchCoarse, NWorldCoarse):
    assert np.all(iPatchCoarse >= 0)
    assert np.all(iPatchCoarse+NPatchCoarse <= NWorldCoarse)
    
    d = np.size(NWorldCoarse)
    Np = np.prod(NPatchCoarse+1)

    def neighboringElements(*index):
        iPatchCoarseRev = iPatchCoarse[::-1].reshape([d] + [1]*d)
        NWorldCoarseRev = NWorldCoarse[::-1].reshape([d] + [1]*d)
        iWorld = iPatchCoarseRev + index
        iWorldNeg = NWorldCoarseRev - iWorld
        lowerCount = np.sum(iWorld==0, axis=0)
        upperCount = np.sum(iWorldNeg==0, axis=0)
        return 2**(d-lowerCount-upperCount)
    
    numNeighboringElements = np.fromfunction(neighboringElements, shape=NPatchCoarse[::-1]+1, dtype='int64').flatten()
    return numNeighboringElements

def tCoordinates(NWorld, iPatch=None, NPatch=None):
    if NPatch is None:
        NPatch = NWorld-1
    else:
        NPatch = NPatch-1

    elementSize = 1./NWorld
    p = pCoordinates(NWorld, iPatch, NPatch)
    t = p + 0.5*elementSize
    return t
                    
def pCoordinates(NWorld, iPatch=None, NPatch=None):
    if iPatch is None:
        iPatch = np.zeros_like(NWorld, dtype='int64')
    if NPatch is None:
        NPatch = NWorld
    d = np.size(iPatch)
    Np = np.prod(NPatch+1)
    p = np.empty((Np,0))
    for k in range(d):
        fk = lambda *index: index[d-k-1]
        newrow = np.fromfunction(fk, shape=NPatch[::-1]+1, dtype='int64').flatten()
        newrow = (iPatch[k]+newrow)/NWorld[k]
        p = np.column_stack([p, newrow])
    return p

def fineIndicesInPatch(NWorldCoarse, NCoarseElement, iPatchCoarse, NPatchCoarse):
    NWorldFine = NCoarseElement*NWorldCoarse

    fineNodeIndexBasis = linearpIndexBasis(NWorldFine)
    fineElementIndexBasis = linearpIndexBasis(NWorldFine-1)
    
    iPatchFine = NCoarseElement*iPatchCoarse

    patchFineNodeIndices = lowerLeftpIndexMap(NPatchCoarse*NCoarseElement, NWorldFine)
    fineNodeStartIndex = np.dot(fineNodeIndexBasis, iPatchFine)
    fineNodeIndices = fineNodeStartIndex + patchFineNodeIndices

    patchFineElementIndices = lowerLeftpIndexMap(NPatchCoarse*NCoarseElement-1, NWorldFine-1)
    fineElementStartIndex = np.dot(fineElementIndexBasis, iPatchFine)
    fineElementIndices = fineElementStartIndex + patchFineElementIndices

    return fineNodeIndices, fineElementIndices

def ignoreDuplicates(row, col, data):
    # Assumes (data, row, col) not in canonical format.
    if len(data) == 0:
        return row, col, data
    order = np.lexsort((row, col))
    row = row[order]
    col = col[order]
    data = data[order]
    unique_mask = ((row[1:] != row[:-1]) |
                   (col[1:] != col[:-1]))
    unique_mask = np.append(True, unique_mask)
    row = row[unique_mask]
    col = col[unique_mask]
    data = data[unique_mask]
    return row, col, data

