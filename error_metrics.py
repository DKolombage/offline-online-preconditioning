import numpy as np

def computeErrorL2(uRef, uApprox, M):
    """
    Compute the relative L2 error:
        || uRef - uApprox ||_M / || uRef ||_M ,  M = mass matrix.

    Parameters
    ----------
    uRef : fine scale reference solution --> ndarray
    uApprox : Approximate solution --> ndarray
    M : Fine-grid mass matrix --> sparse matrix or ndarray

    Returns
    -------
    relerr :  Relative L2 error --> float
    """
    diff = uRef - uApprox
    err = np.sqrt(diff.T @ (M @ diff))
    norm = np.sqrt(uRef.T @ (M @ uRef))
    return err / norm


def computeErrorH1(uRef, uApprox, M, K):
    """
    Compute the relative H1-seminorm error:
        || uRef - uApprox ||_H1 / || uRef ||_H1

    via:
        || v ||_H1^2 = v^T K v + v^T M v

    Parameters
    ----------
    M :  Mass matrix (for L2 term) --> sparse matrix
    K : Stiffness matrix (for gradient term) --> sparse matrix

    Returns
    -------
    relerr :Relative H1-seminorm error --> float
    """
    diff = uRef - uApprox
    err_squared = diff.T @ (K @ diff) + diff.T @ (M @ diff)
    norm_squared = uRef.T @ (K @ uRef) + uRef.T @ (M @ uRef)
    return np.sqrt(err_squared / norm_squared)

def computeErrorH1_Seminorm(uRef, uApprox, K):
    """
    Compute the relative H1-seminorm error:
        || uRef - uApprox ||_H1 / || uRef ||_H1

    via:
        || v ||_H1^2 = v^T K v 

    Parameters
    ----------
    K : Stiffness matrix (for gradient term) --> sparse matrix

    Returns
    -------
    relerr :Relative H1-seminorm error --> float
    """
    diff = uRef - uApprox
    err_squared = diff.T @ (K @ diff)
    norm_squared = uRef.T @ (K @ uRef) 
    return np.sqrt(err_squared / norm_squared)


def computeErrorEnergyNorm(uRef, uApprox, A):
    """
    Compute the relative energy norm error:
        || uRef - uApprox ||_A / || uRef ||_A

    Returns
    -------
    relerr :  Relative energy norm error --> float
    """
    diff = uRef - uApprox
    err = np.sqrt(diff.T @ (A @ diff))
    norm = np.sqrt(uRef.T @ (A @ uRef))
    return err / norm