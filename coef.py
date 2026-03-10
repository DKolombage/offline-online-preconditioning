import numpy as np
import util


def localizeCoefficientFromNodePatch(patch, aFine):
    """
    Extract fine-scale coefficient values (element-wise) on a node-centered patch.

    Parameters
    ----------
    patch : PatchFromNode
        Patch object centered at a coarse node.
    aFine : ndarray
        Global fine-scale coefficient array (flattened over elements).

    Returns
    -------
    aFineLocalized : ndarray
        Localized coefficient restricted to the patch (flattened).
    """
    iPatchWorldCoarse = patch.iPatchWorldCoarse
    NPatchCoarse = patch.NPatchCoarse
    NCoarseElement = patch.world.NCoarseElement
    NPatchFine = NPatchCoarse * NCoarseElement
    iPatchWorldFine = iPatchWorldCoarse * NCoarseElement
    NWorldFine = patch.world.NWorldFine

    # Element-based mapping
    indexMap = util.lowerLeftpIndexMap(NPatchFine - 1, NWorldFine - 1)
    startIndex = util.convertpCoordIndexToLinearIndex(NWorldFine - 1, iPatchWorldFine)
    aFineLocalized = aFine[startIndex + indexMap]

    return aFineLocalized

def localizeCoefficientToCoarseElement(world, aFine, elem_idx):
    """
    Extract the fine-scale coefficient values inside ONE coarse element.

    Parameters
    ----------
    world : World
        Contains NWorldFine, NCoarseElement, etc.
    aFine : ndarray
        Global fine-scale coefficient array (flattened over fine elements).
    elem_idx : tuple(int)
        Coarse element index, e.g. (ex, ey) in 2D.

    Returns
    -------
    aLocal : ndarray
        Local coefficient values on the coarse element, flattened.
        Shape = product(NCoarseElement) in d dimensions.
    """
    NCE = np.asarray(world.NCoarseElement, dtype=int)       # fine elems per coarse elem
    d   = len(NCE)

    # Convert coarse element index to fine-grid starting coordinate
    elem_idx = np.asarray(elem_idx, dtype=int)
    start_fine = elem_idx * NCE

    # Local fine-grid size inside this coarse element
    NLocal = NCE.copy()     # number of fine elements in each dimension

    # Use lower-left element index map (exactly like patch extraction)
    index_map = util.lowerLeftpIndexMap(NLocal - 1, world.NWorldFine - 1)

    start_lin = util.convertpCoordIndexToLinearIndex(world.NWorldFine - 1, start_fine)

    # Extract local coefficients
    aLocal = aFine[start_lin + index_map]

    return aLocal
