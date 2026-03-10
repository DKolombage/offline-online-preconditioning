import numpy as np

import fem, util


class World:
    def __init__(self, NWorldCoarse, NCoarseElement, boundaryConditions = None):
        d = np.size(NWorldCoarse)
        assert(np.size(NCoarseElement) == d)
        if boundaryConditions is None:
            boundaryConditions = np.zeros([d,2], dtype='int64')
        assert(boundaryConditions.shape == (d,2))

        NWorldFine = NWorldCoarse*NCoarseElement

        self.NWorldCoarse = NWorldCoarse
        self.NCoarseElement = NCoarseElement
        self.boundaryConditions = np.array(boundaryConditions)
        self.NWorldFine = NWorldFine
        
        self.NpFine = np.prod(NWorldFine+1)
        self.NtFine = np.prod(NWorldFine)
        self.NpCoarse = np.prod(NWorldCoarse+1)
        self.NtCoarse = np.prod(NWorldCoarse)

    @property
    def localBasis(self):
        if not hasattr(self, '_localBasis'):
            self._localBasis = fem.localBasis(self.NCoarseElement)
        return self._localBasis
    
    @property
    def MLocCoarse(self):
        if not hasattr(self, '_MLocCoarse'):
            self._MLocCoarse = fem.localMassMatrix(self.NWorldCoarse)
        return self._MLocCoarse

    @property
    def MLocFine(self):
        if not hasattr(self, '_MLocFine'):
            self._MLocFine = fem.localMassMatrix(self.NWorldCoarse*self.NCoarseElement)
        return self._MLocFine
    
    @property
    def ALocCoarse(self):
        if not hasattr(self, '_ALocCoarse'):
            self._ALocCoarse = fem.localStiffnessMatrix(self.NWorldCoarse)
        return self._ALocCoarse

    @property
    def ALocFine(self):
        if not hasattr(self, '_ALocFine'):
            self._ALocFine = fem.localStiffnessMatrix(self.NWorldCoarse*self.NCoarseElement)
        return self._ALocFine
    
    @property
    def ALocMatrixCoarse(self):
        if not hasattr(self, '_ALocMatrixCoarse'):
            self._ALocMatrixCoarse = fem.localStiffnessTensorMatrixCoefficient(self.NWorldCoarse)
        return self._ALocMatrixCoarse
        
    @property
    def ALocMatrixFine(self):
        if not hasattr(self, '_ALocMatrixFine'):
            self._ALocMatrixFine = fem.localStiffnessTensorMatrixCoefficient(self.NWorldCoarse*self.NCoarseElement)
        return self._ALocMatrixFine
    
    @property
    def FLocCoarse(self):
        if not hasattr(self, '_FLocCoarse'):
            self._FLocCoarse = fem.localFaceMassMatrix(self.NWorldCoarse)
        return self._FLocCoarse

    @property
    def FLocFine(self):
        if not hasattr(self, '_FLocFine'):
            self._FLocFine = fem.localFaceMassMatrix(self.NWorldCoarse*self.NCoarseElement)
        return self._FLocFine

class PatchFromNode:
    """
    Patch centered around a coarse node, for use in node-based LOD.
    Constructs a patch of coarse elements of size (2k x 2k) around the node.

    Parameters
    ----------
    world : World
        The global world object containing coarse/fine mesh information.
    k : int
        The patch radius (in coarse elements).
    nodeInd : int
        Linear index of the central coarse node.
    """

    def __init__(self, world, k, nodeInd):
        self.world = world
        self.k = k
        self.nodeInd = nodeInd  # index of the central coarse node

        d = np.size(world.NWorldCoarse)
        NWorldNodes = world.NWorldCoarse + 1  # total coarse nodes in each dimension

        # Convert linear node index to multi-index
        iNodeWorld = util.convertpLinearIndexToCoordIndex(NWorldNodes, nodeInd)
        self.iNodeWorld = iNodeWorld  # shape: (d,)

        # Compute patch start and end in coarse ELEMENT indices
        iPatchStartElem = iNodeWorld - k
        iPatchEndElem   = iNodeWorld + k  # not inclusive

        # Check if patch lies fully inside domain
        if np.any(iPatchStartElem < 0) or np.any(iPatchEndElem > world.NWorldCoarse):
            raise ValueError(f"Patch for node {nodeInd} exceeds domain boundary.")

        # Store patch location and shape
        self.iPatchWorldCoarse = iPatchStartElem
        self.NPatchCoarse = 2 * k * np.ones_like(iNodeWorld, dtype=int)
        self.iElementPatchCoarse = iNodeWorld - iPatchStartElem

        # Convert patch size to fine scale
        self.NPatchFine = self.NPatchCoarse * world.NCoarseElement

        # Compute number of fine nodes and elements
        self.NpFine = np.prod(self.NPatchFine + 1)
        self.NtFine = np.prod(self.NPatchFine)

        # Also store coarse grid counts (optional)
        self.NpCoarse = np.prod(self.NPatchCoarse + 1)
        self.NtCoarse = np.prod(self.NPatchCoarse)
    def __repr__(self):
        return (f"PatchFromNode(nodeInd={self.nodeInd}, "
                f"iNodeWorld={self.iNodeWorld}, "
                f"iPatchStartElem={self.iPatchWorldCoarse}, "
                f"NPatchCoarse={self.NPatchCoarse}, "
                f"NPatchFine={self.NPatchFine})")

def construct_center_node_patch(world, k):
    """
    Construct a patch centered around the central node of the coarse grid.

    Parameters
    ----------
    world : World
        The global mesh information (must be initialized already).
    k : int
        Patch radius (in coarse elements).

    Returns
    -------
    PatchFromNode
        A patch object centered at the central coarse node.
    """
    # Get the shape of the coarse node grid
    NWorldNodes = world.NWorldCoarse + 1

    # Find center node in each direction (floor division)
    center_node_coords = NWorldNodes // 2  # e.g., [4,4] if shape is [9,9]

    # Convert coordinate to linear index
    basis = util.linearpIndexBasis(NWorldNodes)
    center_node_index = np.dot(basis, center_node_coords)
    print("center_node_index",center_node_index)

    # Construct the patch around this node
    patch = PatchFromNode(world, k, nodeInd=center_node_index)

    return patch


