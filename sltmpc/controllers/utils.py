import cvxpy as cp
import numpy as np

from ampyc.utils import Polytope


def build_system_responses(structure: str, N: int, n: int, m: int, n_x: int, n_u: int, n_w: int) -> tuple[cp.Expression, cp.Expression, cp.Expression, cp.Expression]:
    """
    Builds the system responses and corresponding dual variables for various SLTMPC controllers.

    Args:
        structure (str): The structure of the system responses, either "Toeplitz" or "unstructured".
        N (int): The prediction horizon.
        n (int): The state dimension.
        m (int): The control input dimension.
        n_x (int): The dimension of the state constraints.
        n_u (int): The dimension of the control constraints.
        n_w (int): The dimension of the disturbance.

    Returns:
        tuple: A tuple containing the system responses & dual variables:
            - Phi_e: The state system response.
            - Phi_k: The input system response.
            - Lambda_x: The dual variable for the state constraints.
            - Lambda_u: The dual variable for the input constraints.
    """

    if structure in ["Toeplitz"]:
        # Toeplitz structured system responses
        Phi_e = np.eye(N*n)
        Lambda_x = cp.kron(np.eye(N), cp.Variable((n_x, n_w), nonneg=True))
        for i in range(N-1):
            Phi_e = Phi_e + cp.kron(np.eye(N,k=-i-1), cp.Variable((n,n)))
            Lambda_x = Lambda_x + cp.kron(np.eye(N,k=-i-1), cp.Variable((n_x, n_w), nonneg=True))

        Phi_k = 0
        Lambda_u = 0
        for i in range(N):
            Phi_k = Phi_k + cp.kron(np.eye(N,k=-i), cp.Variable((m,n)))
            Lambda_u = Lambda_u + cp.kron(np.eye(N,k=-i), cp.Variable((n_u, n_w), nonneg=True))
            
    elif structure in ["unstructured"]:
        # unstructured system responses
        Phi_e = []
        Phi_k = []
        Lambda_x = []
        Lambda_u = []
        for i in range(N):
            if i == 0:
                row_e = [cp.Variable((n,n)), np.zeros((n,n*(N-1)))]
                row_k = [cp.Variable((m,n)), np.zeros((m,n*(N-1)))]
                row_lam_x = [cp.Variable((n_x,n_w), nonneg=True), np.zeros((n_x,n_w*(N-1)))]
                row_lam_u = [cp.Variable((n_u,n_w), nonneg=True), np.zeros((n_u,n_w*(N-1)))]
            elif i == N-1:
                row_e = [cp.Variable((n,n*N))]
                row_k = [cp.Variable((m,n*N))]
                row_lam_x = [cp.Variable((n_x,n_w*N), nonneg=True)]
                row_lam_u = [cp.Variable((n_u,n_w*N), nonneg=True)]
            else:
                row_e = [cp.Variable((n,n*(i+1))), np.zeros((n,n*(N-1-i)))]
                row_k = [cp.Variable((m,n*(i+1))), np.zeros((m,n*(N-1-i)))]
                row_lam_x = [cp.Variable((n_x,n_w*(i+1)), nonneg=True), np.zeros((n_x,n_w*(N-1-i)))]
                row_lam_u = [cp.Variable((n_u,n_w*(i+1)), nonneg=True), np.zeros((n_u,n_w*(N-1-i)))]

            Phi_e.append(row_e)    
            Phi_k.append(row_k)
            Lambda_x.append(row_lam_x)    
            Lambda_u.append(row_lam_u)

        Phi_e = cp.bmat(Phi_e)
        Phi_k = cp.bmat(Phi_k)
        Lambda_x = cp.bmat(Lambda_x)
        Lambda_u = cp.bmat(Lambda_u)

    else:
        raise ValueError(f"Unsupported value for structure: {structure}\nsupported values are [Toeplitz, unstructured]")

    return Phi_e, Phi_k, Lambda_x, Lambda_u


def build_filter(structure: str, N: int, n: int, n_w: int, n_D: int) -> tuple[cp.Expression, cp.Expression]:
    """
    Builds the filter for filter-based SLTMPC controllers.

    Args:
        structure (str): The structure of the filter's diagonal, either "scalar" or "diagonal".
        N (int): The prediction horizon.
        n (int): The state dimension.

    Returns:
        tuple: A tuple containing the filter variables:
            - Sigma_diag: The diagonal elements of the filter matrix.
            - Sigma: The filter matrix without the diagonal elements.
    """

    if structure in ["diagonal"]:
        sigma = cp.Variable((N,n))
        Sigma = []
        Sigma_diag = []
        for i in range(N):
            if i == 0:
                row_sig = [np.zeros((n,n)), np.zeros((n,n*(N-1)))]
                row_sig_diag = [cp.diag(sigma[i]), np.zeros((n,n*(N-1)))]
            elif i == N-1:
                row_sig = [cp.Variable((n,n*(N-1))), np.zeros((n,n))]
                row_sig_diag = [np.zeros((n,n*(N-1))), cp.diag(sigma[i])]
            else:
                row_sig = [cp.Variable((n,n*i)), np.zeros((n,n*(N-i)))]
                row_sig_diag = [np.zeros((n,n*i)), cp.diag(sigma[i]), np.zeros((n,n*(N-1-i)))]

            Sigma.append(row_sig) 
            Sigma_diag.append(row_sig_diag)

        Sigma = cp.bmat(Sigma)
        Sigma_diag = cp.bmat(Sigma_diag)
            
    elif structure in ["scalar"]:
        sigma = cp.Variable((N,))
        Sigma = []
        Sigma_diag = []
        for i in range(N):
            if i == 0:
                row_sig = [np.zeros((n,n)), np.zeros((n,n*(N-1)))]
                row_sig_diag = [sigma[i]*np.eye(n), np.zeros((n,n*(N-1)))]
            elif i == N-1:
                row_sig = [cp.Variable((n,n*(N-1))), np.zeros((n,n))]
                row_sig_diag = [np.zeros((n,n*(N-1))), sigma[i]*np.eye(n)]
            else:
                row_sig = [cp.Variable((n,n*i)), np.zeros((n,n*(N-i)))]
                row_sig_diag = [np.zeros((n,n*i)), sigma[i]*np.eye(n), np.zeros((n,n*(N-1-i)))]

            Sigma.append(row_sig) 
            Sigma_diag.append(row_sig_diag)

        Sigma = cp.bmat(Sigma)
        Sigma_diag = cp.bmat(Sigma_diag)

    else:
        raise ValueError(f"Unsupported value for structure: {structure}\nsupported values are [scalar, diagonal]")

    Lambda_w = []
    for _ in range(n_D):
        Lambda_d = []
        for i in range(N):
            if i == 0:
                row_lam_d = [cp.Variable((n_w,n_w), nonneg=True), np.zeros((n_w,n_w*(N-1)))]
            elif i == N-1:
                row_lam_d = [cp.Variable((n_w,n_w*N), nonneg=True)]
            else:
                row_lam_d = [cp.Variable((n_w,n_w*(i+1)), nonneg=True), np.zeros((n_w,n_w*(N-1-i)))]
            Lambda_d.append(row_lam_d) 

        Lambda_w.append(cp.bmat(Lambda_d))

    return Sigma, Sigma_diag, sigma, Lambda_w


def compute_sldrs(Phi_x: np.ndarray, W: Polytope, Toeplitz: bool = False) -> list[Polytope]:
    """
    Computes the system level disturbance reachable sets (SL-DRS) for a given system response
    matrix and disturbance polytope.

    For more information on SL-DRS, see Section 3.4 of:
    J. Sieber, "A System Level Robust Model Predictive Control Framework with Asynchronous
    Computations", PhD thesis, ETH Zurich, 2024.

    Args:
        Phi_x (np.ndarray): The state system response.
        W (Polytope): The disturbance polytope.
        Toeplitz (bool): Whether the system responses are structured as Toeplitz matrices or not.
    Returns:
        list[Polytope]: A list of Polytope objects representing the SL-DRS for each time step.
    """

    # get dimensions
    n = W.dim
    N = int(Phi_x.shape[0]/n)

    # compute SL-DRS
    if Toeplitz:
        F = [W]
        for i in range(1, N):
            F.append(Phi_x[i*n:(i+1)*n,:n]@W + F[-1])
    else:
        F = []
        for i in range(N):
            temp = Phi_x[i*n:(i+1)*n,:n]@W
            for j in range(1,i+1):
                temp += Phi_x[i*n:(i+1)*n,j*n:(j+1)*n]@W
            F.append(temp)
    
    return F
