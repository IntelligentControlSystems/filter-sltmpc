import cvxpy as cp
import numpy as np
from scipy.linalg import block_diag, sqrtm
from itertools import product

from ampyc.controllers import ControllerBase

from sltmpc.controllers import build_system_responses, build_filter
from sltmpc.utils import compute_mrci

class SLS_MPC(ControllerBase):
    '''
    Implements the system level synthesis MPC (SLS-MPC) controller proposed in:

    S. Chen, V. M. Preciado, M. Morari, and N. Matni, "Robust Model Predictive Control
    with Polytopic Model Uncertainty through System Level Synthesis", Automatica, 2024.
    
    Optional Arguments:
        - X_f (Polytope), P_f (np.ndarray): terminal set and terminal cost matrix. If any of them
            is not provided, the state cost is used as the terminal cost, and the terminal set is computed
            as the maximum RCI set.
        - regularize (callable): Function defining the regularization term in the cost. It takes the system
            responses and filter variables as arguments and returns a regularization term (default: lambda x: 0.0).
        - timing (bool): Whether to return the solve time of the solver (default: False).
    '''

    def _init_problem(self, sys, params, *args, **kwargs):
        # handle arguments
        self.X_f = kwargs.get('X_f', None)
        self.P_f = kwargs.get('P_f', None)
        self.regularize = kwargs.get('regularize', lambda x: 0.0)

        # look up control parameters & dimensions
        Q, R, N = (params.Q, params.R, params.N)
        n, m = (sys.n, sys.m)

        # look up system matrices
        A, B = (sys.A, sys.B)

        # compute terminal ingredients if not provided
        if any(element is None for element in [self.X_f, self.P_f]):
            # set the terminal cost
            self.P_f = Q

            # compute the MRCI terminal set
            self.X_f = compute_mrci(sys)

        # look up system constraints & disturbance set
        H_x, h_x = (sys.X.A, sys.X.b.reshape(-1,1))
        H_u, h_u = (sys.U.A, sys.U.b.reshape(-1,1))
        H_f, h_f = (self.X_f.A, self.X_f.b.reshape(-1,1))
        n_x, n_u = (h_x.shape[0], h_u.shape[0])
        
        # process model uncertainty into a single list
        if hasattr(sys, 'Delta_A'):
            if hasattr(sys, 'Delta_B'):
                D = list(product(sys.Delta_A, sys.Delta_B))
            else:
                D = sys.Delta_A
        else:
            if hasattr(sys, 'Delta_B'):
                D = sys.Delta_B
            else:
                D = []
        n_D = len(D)

        # build condensed dynamics
        ZA_N1 = np.kron(np.eye(N+1,k=-1), A)
        ZB_N1 = np.kron(np.eye(N+1,k=-1), B)
        ZA = np.kron(np.eye(N,k=-1), A)
        ZB = np.kron(np.eye(N,k=-1), B)

        # define the optimization variables
        self.x_0 = cp.Parameter((n, 1))
        self.z = cp.Variable((n*(N+1), 1))
        self.v = cp.Variable((m*(N+1), 1)) # the last control input is not used, but needed for the condensed formulation
        self.p = cp.Variable((n*N, 1))
        # get system responses; note we don't need the dual variables here
        self.Phi_x, self.Phi_u, _, _ = build_system_responses("unstructured", N, n, m, n_x, n_u, sys.W.b.size)
        # get filter; note we don't need the dual variables here
        self.Sigma, self.Sigma_diag, self.sigma, _ = build_filter("diagonal", N, n, sys.W.b.size, n_D)

        ### define the objective [(40) in the paper]

        # quadratic nominal cost
        C = block_diag(*[np.kron(np.eye(N), sqrtm(Q)), sqrtm(self.P_f), np.kron(np.eye(N+1), sqrtm(R))])
        nominal_obj = cp.square(cp.norm(C@cp.vstack([self.z, self.v]), 2))

        # regularization term
        regularizer = self.regularize([self.Phi_x, self.Phi_u, self.Sigma, self.Sigma_diag])

        # objective
        objective = nominal_obj + regularizer

        ###

        ### define the constraints

        # dynamics [(26) in the paper]
        constraints = [self.Phi_x - ZA@self.Phi_x - ZB@self.Phi_u == self.Sigma + self.Sigma_diag]
        constraints.append(ZA_N1@self.z + ZB_N1@self.v + cp.vstack([self.x_0, self.p]) == self.z)
      
        # tightened state constraints [(37) in the paper]
        for (f,b) in zip(H_x, h_x):
            for i in range(N):
                if i == 0:
                    constraints.append(f@self.z[i*n:(i+1)*n] <= b)
                else:
                    constraints.append(f@self.z[i*n:(i+1)*n] + cp.norm(f@self.Phi_x[(i-1)*n:(i)*n, :], 1) <= b)
        
        # tightened terminal constraints [(38) in the paper]
        for (f,b) in zip(H_f, h_f):
            constraints.append(f@self.z[-n:] + cp.norm(f@self.Phi_x[-n:, :], 1) <= b)

        # tightened input constraints [(39) in the paper]
        for (f,b) in zip(H_u, h_u):
            for i in range(N):
                if i == 0:
                    constraints.append(f@self.v[i*m:(i+1)*m] <= b)
                else:
                    constraints.append(f@self.v[i*m:(i+1)*m] + cp.norm(f@self.Phi_u[(i-1)*m:(i)*m, :], 1) <= b)

        # disturbance overapproximation [(36) in the paper]
        E = np.eye(n)
        sig_w = []
        for e in E:
            sig_w.append(max(sys.W.support(e), sys.W.support(-e)))
        
        for d in range(n_D):
            # get disturbance vertex
            dA, dB = D[d]

            # build the disturbance auxiliary variables
            psi = np.kron(np.eye(N), dA)@self.z[:-n] + np.kron(np.eye(N), dB)@self.v[:-m] - self.p
            Psi = np.kron(np.eye(N,k=-1), dA)@self.Phi_x + np.kron(np.eye(N,k=-1), dB)@self.Phi_u - self.Sigma
            # add constraints
            for j in range(n):
                e = E[j,:]
                sig_w_j = sig_w[j]
                for i in range(N):
                    constraints.append(cp.abs(e@psi[i*n:(i+1)*n]) + sig_w_j + cp.norm(e@Psi[i*n:(i+1)*n, :], 1) <= self.sigma[i, j])

        ###

        # define the CVX optimization problem object
        self.prob = cp.Problem(cp.Minimize(objective), constraints)

    def _define_output_mapping(self):
        return_dict = {
            'control': self.v,
            'state': self.z,
            'p': self.p,
            'Phi_x': self.Phi_x,
            'Phi_u': self.Phi_u,
            'Sigma': self.Sigma + self.Sigma_diag,
        }

        return return_dict

class RCI_Controller(ControllerBase):
    '''
    Implements a controller for a given RCI set. This controller is used for a shrinking horizon
    strategy in SLS-MPC. For more details, see:

    Bujarbaruah et al., "Robust MPC for LPV systems via a novel optimization-based constraint
    tightening", Automatica, 2022.

    Arguments:
        - X_f (Polytope): The RCI set to be used as the terminal set.
    '''

    def _init_problem(self, sys, params, *args, **kwargs):
        # handle arguments
        self.X_f = kwargs.get('X_f', None)
        if self.X_f is None:
            raise ValueError("The terminal set X_f must be provided to the MCI controller.")

        # look up control parameters & dimensions
        Q, R = (params.Q, params.R)
        n, m = (sys.n, sys.m)

        # look up system matrices
        A, B = (sys.A, sys.B)

        # look up system constraints & disturbance set
        H_u, h_u = (sys.U.A, sys.U.b.reshape(-1,1))
        H_f, h_f = (self.X_f.A, self.X_f.b.reshape(-1,1))
        
        # process model uncertainty into a single list
        if hasattr(sys, 'Delta_A'):
            if hasattr(sys, 'Delta_B'):
                D = list(product(sys.Delta_A, sys.Delta_B))
            else:
                D = sys.Delta_A
        else:
            if hasattr(sys, 'Delta_B'):
                D = sys.Delta_B
            else:
                D = []
        n_D = len(D)

        # define the optimization variables
        self.x_0 = cp.Parameter((n, 1))
        self.u = cp.Variable((m, 1))

        # allocate constraints and objective
        constraints = []
        objective = 0.0

        # define state constraints & objective
        for w_v in sys.W.V:        
            for d in range(n_D):
                # get disturbance vertex
                dA, dB = D[d]

                # compute uncertain system evolution
                x1 = (A+dA)@self.x_0 + (B+dB)@self.u + w_v.reshape(-1,1)

                # ensure containment in the terminal set & compute objective
                constraints.append(H_f@x1 <= h_f)
                objective += cp.quad_form(x1, Q)
        
        # input constraints
        constraints.append(H_u@self.u <= h_u)
        objective += cp.quad_form(self.u, R)

        ###

        # define the CVX optimization problem object
        self.prob = cp.Problem(cp.Minimize(objective), constraints)

    def _define_output_mapping(self):
        return_dict = {
            'control': self.u,
            'state': self.x_0
        }

        return return_dict