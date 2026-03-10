import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import util , fem

np.random.seed(5)
def to_bmap(marker, d):
    """
    Normalize a boundary marker to boolean map of shape (d,2):
    True  => Dirichlet on that side
    False => not Dirichlet
    """
    if marker is None:
        return np.ones((d, 2), dtype=bool)          # Dirichlet on all sides
    marker = np.asarray(marker)
    if marker.dtype == bool:
        assert marker.shape == (d, 2)
        return marker
    # legacy int markers: 0 => Dirichlet, 1 => free
    assert marker.shape == (d, 2)
    return (marker == 0)

def free_fixed_from_N(N, boundaryMarker=None):
    """
    From a per-dimension cell-count vector N (e.g. world.NWorldFine),
    return (free, fixed) node-index arrays using the global boolean convention.
    """
    N = np.asarray(N, dtype=int)
    bmap = to_bmap(boundaryMarker, len(N))
    Np = int(np.prod(N + 1))
    fixed = util.boundarypIndexMap(N, boundaryMap=bmap)
    free  = np.setdiff1d(np.arange(Np), fixed)
    return free, fixed

def solve_fem_standard(world, aFine, f_func, boundaryMarker=None):
    """
    Solve -div(a grad u) = f on the fine mesh (homog. Dirichlet on marked sides).

    boundaryMarker: (d,2) boolean or int; True (or 0) => Dirichlet.
    Returns:
      uFineFull, AFineFull, AFineFree, MFineFull, MFineFree, bFineFull, bFineFree
    """
    Nf = np.asarray(world.NWorldFine, dtype=int)
    NpFine = int(np.prod(Nf + 1))
    d = len(Nf)

    # free/fixed nodes on fine grid
    freeFine, fixedFine = free_fixed_from_N(Nf, boundaryMarker)

    # assemble
    AFine = fem.assemblePatchMatrix(Nf, world.ALocFine, aFine)
    MFine = fem.assemblePatchMatrix(Nf, world.MLocFine)

    # sample RHS at nodes
    coords_fine = util.pCoordinates(Nf)
    f_vals = f_func(coords_fine).ravel()

    # build b = M f
    bFine = MFine @ f_vals

    # restrict to free
    AFineFree = AFine[freeFine][:, freeFine]
    MFineFree = MFine[freeFine][:, freeFine]
    bFineFree = bFine[freeFine]

    # solve A_ff u_f = b_f
    uFineFree = spla.spsolve(AFineFree.tocsc(), bFineFree)

    # embed selector E: R^{n_free} -> R^{NpFine}
    E = sp.csr_matrix(
        (np.ones(len(freeFine)), (freeFine, np.arange(len(freeFine)))),
        shape=(NpFine, len(freeFine))
    )

    # “full with Dirichlet applied” (zeros on boundary rows/cols) – storage/visualization
    uFineFull = E @ uFineFree
    AFineFull = (E @ AFineFree @ E.T).tocsr()
    MFineFull = (E @ MFineFree @ E.T).tocsr()
    bFineFull = E @ bFineFree

    return uFineFull,uFineFree, AFineFull, AFineFree, MFineFull, MFineFree, bFineFull, bFineFree, freeFine


def plot_errors(u, u_ref, NFine):
    """
    Plot error between u and u_ref for 1D or 2D fine grid.
    """
    diff = u - u_ref
    err_norm = np.linalg.norm(diff)
    print(f"L2 Error Norm: {err_norm:.3e}")

    d = len(NFine)

    if d == 1:
        Nx = NFine[0]
        x_grid = np.linspace(0, 1, Nx + 1)

        plt.figure(figsize=(6, 4))
        plt.plot(x_grid, u, label='Computed', linestyle='-')
        plt.plot(x_grid, u_ref, label='Reference', linestyle='--')
        plt.plot(x_grid, diff, label='Error', linestyle=':')
        plt.title('1D Solution and Error')
        plt.xlabel('x')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    elif d == 2:
        Nx, Ny = NFine
        uGrid = (u - u_ref).reshape((Ny + 1, Nx + 1))

        plt.figure(figsize=(6, 5))
        plt.imshow(uGrid, origin='lower', cmap='viridis')
        plt.title(f'2D Error ||u - u_ref|| = {err_norm:.2e}')
        plt.colorbar(label='Error')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.tight_layout()
        plt.show()

    else:
        raise NotImplementedError("plot_errors only supports 1D and 2D.")

