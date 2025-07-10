import cvxpy as cp
import numpy as np
from scipy.linalg import block_diag, sqrtm
from itertools import product

from ampyc.controllers import ControllerBase
from ampyc.utils import Polytope, compute_mrpi, LQR

from sltmpc.controllers import build_system_responses, build_filter


class Filter_SLTMPC(ControllerBase):
    '''
    Implements the filter-based system level tube-MPC (SLTMPC) controllers proposed in
    Section 4.3 (Eq. 4.19) and 4.4 (Eq. 4.25) of:

    J. Sieber, "A System Level Robust Model Predictive Control Framework with
    Asynchronous Computations", PhD thesis, ETH Zurich, 2024.
    
    Optional Arguments:
        - terminal_eq (bool): Whether to use a terminal equality constraint (default: False). If True,
            the terminal set is not used, and the last state is constrained to be zero (i.e., the origin).
            Only use this option for open-loop simulations!
        - W_tilde (Polytope): Auxiliary disturbance set used for uncertainty overapproximation in the filter.
            If not provided, the disturbance set of the system is used (default: sys.W).
        - structure (str): Structure of the filter diagonal. Supported values are "diagonal" and "scalar".
            See Remarks 4.6 & 4.9 in the thesis for more details (default: "scalar").
        - scale_terminal_set (bool): Whether to scale the terminal set or not (default: True). This option
            essentially switches between Eq. (4.19) and Eq. (4.25) in the thesis. True uses the recursively
            feasible version (4.25), while False uses the original formulation (4.19).
        - X_f (Polytope), K_f (np.ndarray), P_f (np.ndarray): terminal set, terminal controller, and
            terminal cost matrix. If any of them is not provided, ALL are computed using the LQR controller.
        - regularize (callable): Function defining the regularization term in the cost. It takes the system
            responses, the filter variables, and dual variables as arguments and returns a regularization
            term (default: lambda x: 0.0).
        - timing (bool): Whether to return the solve time of the solver (default: False).
    
    Note:
        These controllers were originally proposed in:
        J. Sieber, A. Didier, and M. N. Zeiligner, "Computationally efficient system level tube-MPC for
        uncertain systems", Automatica, 2025.
    '''

    def _init_problem(self, sys, params, *args, **kwargs):
        # handle arguments
        self.terminal_eq = kwargs.get('terminal_eq', False)
        self.W_til = kwargs.get('W_tilde', sys.W) # if no W_tilde is provided, use the disturbance set
        self.structure = kwargs.get('structure', "scalar")  # structure of the filter diagonal
        self.scale_terminal_set = kwargs.get('scale_terminal_set', True) # use (4.25) or (4.19) in the thesis
        self.X_f = kwargs.get('X_f', None)
        self.K_f = kwargs.get('K_f', None)
        self.P_f = kwargs.get('P_f', None)
        self.regularize = kwargs.get('regularize', lambda x: 0.0)

        # look up control parameters & dimensions
        Q, R, N = (params.Q, params.R, params.N)
        n, m = (sys.n, sys.m)

        # look up system matrices
        A, B = (sys.A, sys.B)

        # look up system constraints & disturbance set
        H_x, h_x = (sys.X.A, sys.X.b.reshape(-1,1))
        H_u, h_u = (sys.U.A, sys.U.b.reshape(-1,1))
        H_w, h_w = (sys.W.A, sys.W.b.reshape(-1,1))
        H_w_til, h_w_til = (self.W_til.A, self.W_til.b.reshape(-1,1))
        n_x, n_u, n_w_til = (h_x.shape[0], h_u.shape[0], h_w_til.shape[0])

        # compute support function of W w.r.t. W_tilde; this is needed for the disturbance overapproximation
        # more details are provided in Appendix 4.C of the thesis
        if self.W_til == sys.W:
            W_til_support = self.W_til.b.reshape(-1, 1)
        else:
            W_til_support = np.vstack([sys.W.support(eta) for eta in H_w_til])

        # compute terminal ingredients if not provided
        if any(element is None for element in [self.X_f, self.K_f, self.P_f]):
            if self.terminal_eq:
                self.P_f = np.zeros((n, n))  # terminal cost is zero if terminal equality constraint is used
            else:
                # compute the LQR controller & cost
                self.K_f, self.P_f = LQR(A, B, Q, R)

                # compute the MRPI terminal set
                Omega = Polytope(np.vstack([sys.X.A, sys.U.A @ self.K_f]), np.hstack([sys.X.b, sys.U.b]))
                self.X_f = compute_mrpi(A + B @ self.K_f, Omega, sys.W)
        
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
        # get system responses
        self.Phi_e, self.Phi_k, Lambda_x, Lambda_u = build_system_responses("unstructured", N, n, m, n_x, n_u, n_w_til)
        # get filter
        self.Sigma, self.Sigma_diag, self.sigma, Lambda_w = build_filter(self.structure, N, n, n_w_til, n_D)

        ### define the objective [(4.25a) in the thesis]

        # quadratic nominal cost
        C = block_diag(*[np.kron(np.eye(N), sqrtm(Q)), sqrtm(self.P_f), np.kron(np.eye(N+1), sqrtm(R))])
        nominal_obj = cp.square(cp.norm(C@cp.vstack([self.z, self.v]), 2))

        # regularization term
        regularizer = self.regularize([self.Phi_e, self.Phi_k, self.Sigma, self.Sigma_diag, Lambda_x, Lambda_u])

        # objective
        objective = nominal_obj + regularizer

        ###

        ### define the constraints

        # dynamics [(4.25c & d) in the thesis]
        constraints = [self.Phi_e - ZA@self.Phi_e - ZB@self.Phi_k == self.Sigma + self.Sigma_diag]
        constraints.append(ZA_N1@self.z + ZB_N1@self.v + cp.vstack([self.x_0, self.p]) == self.z)
        
        # state constraints [(4.25f) in the thesis]
        constraints.append(np.kron(np.eye(N+1), H_x)@self.z <= np.kron(np.ones((N+1,1)), h_x)
                           - cp.vstack([np.zeros((n_x,1)), Lambda_x@np.kron(np.ones((N,1)), h_w_til)]))
        constraints.append(Lambda_x@np.kron(np.eye(N), H_w_til) == np.kron(np.eye(N), H_x)@self.Phi_e)

        # input constraints [(4.25g) in the thesis]
        constraints.append(np.kron(np.eye(N+1), H_u)@self.v <= np.kron(np.ones((N+1,1)), h_u)
                           - cp.vstack([np.zeros((n_u,1)), Lambda_u@np.kron(np.ones((N,1)), h_w_til)]))
        constraints.append(Lambda_u@np.kron(np.eye(N), H_w_til) == np.kron(np.eye(N), H_u)@self.Phi_k)

        # disturbance overapproximation [(4.25k) in the thesis]
        for d in range(n_D):
            # get disturbance vertex
            dA, dB = D[d]

            # build the disturbance auxiliary variables [(4.16) in the thesis]
            psi = np.kron(np.eye(N), dA)@self.z[:-n] + np.kron(np.eye(N), dB)@self.v[:-m] - self.p
            Psi = np.kron(np.eye(N,k=-1), dA)@self.Phi_e + np.kron(np.eye(N,k=-1), dB)@self.Phi_k - self.Sigma

            # add constraints
            if self.structure in ["diagonal"]:
                constraints.append(np.kron(np.eye(N), H_w_til)@psi <= cp.vstack([cp.kron(cp.diag(self.sigma[i]), np.eye(n))@h_w_til for i in range(N)])
                                - np.kron(np.ones((N,1)), W_til_support)
                                - cp.vstack([np.zeros((n_w_til,1)), Lambda_w[d][n_w_til:]@np.kron(np.ones((N,1)), h_w_til)]))
            elif self.structure in ["scalar"]:
                constraints.append(np.kron(np.eye(N), H_w_til)@psi <= cp.vstack([self.sigma[i]*h_w_til for i in range(N)])
                                - np.kron(np.ones((N,1)), W_til_support)
                                - cp.vstack([np.zeros((n_w_til,1)), Lambda_w[d][n_w_til:]@np.kron(np.ones((N,1)), h_w_til)]))
            else:
                raise ValueError(f"Unknown filter structure: {self.structure}. Supported structures are 'diagonal' and 'scalar'.")
            
            constraints.append(Lambda_w[d]@np.kron(np.eye(N), H_w_til) == np.kron(np.eye(N), H_w_til)@Psi)

        ###

        ### define the terminal set constraints

        if self.terminal_eq:

            print("Using terminal equality constraint!")

            # override scaling flag
            self.scale_terminal_set = False

            # use a terminal equality constraint (only use this for open-loop simulations)
            constraints.append(self.z[-n:] == np.zeros((n,1)))

        else:
            if self.scale_terminal_set:
                """
                If True, we implement recursively feasible SLTMPC (4.25) in the thesis, which uses
                a scaled terminal set. The implementation details are found in Appendix 3.B (Lemma 3.27)
                and Appendix 4.C (Lemma 4.20) of the thesis.
                """

                print("Using scaled terminal set!")

                # define terminal filter variables
                self.Xi = cp.Variable((n,n*N))

                # define scaling variables
                self.alpha = cp.Variable((1,1), nonneg=True)
                self.Gamma = cp.Variable((n, n))

                # define terminal scaling dual variables
                Lambda_xf = cp.Variable((n_x, self.X_f.b.shape[0]), nonneg=True)
                Lambda_uf = cp.Variable((n_u, self.X_f.b.shape[0]), nonneg=True)
                Lambda_X_f_w = cp.Variable((self.X_f.b.shape[0], n_w_til), nonneg=True)
                Lambda_X_f = cp.Variable((self.X_f.b.shape[0], self.X_f.b.shape[0]), nonneg=True)

                # define terminal overapproximation dual variables
                Lambda_wf = [cp.Variable((n_w_til, N*n_w_til), nonneg=True) for _ in range(n_D)]
                Lambda_hf = [cp.Variable((n_w_til, self.X_f.b.shape[0]), nonneg=True) for _ in range(n_D)]

                # define terminal overapproximation constraint [(4.25l) in the thesis]
                for d in range(n_D):
                    # get disturbance vertex
                    dA, dB = D[d]

                    # build the terminal disturbance auxiliary variable
                    Psi = dA@self.Phi_e[-n:,:] + dB@self.Phi_k[-m:,:] - self.Xi

                    # add constraints
                    if self.structure in ["diagonal"]:
                        constraints.append(Lambda_hf[d]@self.X_f.b.reshape(-1,1) <= cp.kron(cp.diag(self.sigma[-1]), np.eye(n))@h_w_til
                                        - W_til_support - Lambda_wf[d]@np.kron(np.ones((N,1)), h_w_til))
                    elif self.structure in ["scalar"]:
                        constraints.append(Lambda_hf[d]@self.X_f.b.reshape(-1,1) <= self.sigma[-1]*h_w_til
                                        - W_til_support - Lambda_wf[d]@np.kron(np.ones((N,1)), h_w_til))
                    else:
                        raise ValueError(f"Unknown filter structure: {self.structure}. Supported structures are 'diagonal' and 'scalar'.")
                    
                    constraints.append(Lambda_wf[d]@np.kron(np.eye(N), H_w_til) == H_w_til@Psi)
                    constraints.append(Lambda_hf[d]@self.X_f.A == self.alpha*H_w_til@(dA + dB@self.K_f))

                # quasi Toeplitz constraint for last row in system responses [(4.25e) in the thesis]
                constraints.append(self.Phi_e[-n:,:-n] == A@self.Phi_e[-n:,n:] + B@self.Phi_k[-m:,n:] + self.Xi[:, n:])
                # define disturbance strength after horizon
                constraints.append(self.Gamma == A@self.Phi_e[-n:,:n] + B@self.Phi_k[-m:,:n] + self.Xi[:, :n])

                # scale terminal set [(4.25j) in the thesis]
                constraints.append(Lambda_X_f_w@H_w_til == self.X_f.A@self.Gamma)
                constraints.append(Lambda_X_f@self.X_f.A == self.alpha*self.X_f.A@(A + B@self.K_f))
                constraints.append(Lambda_X_f@self.X_f.b.reshape(-1,1) <= self.alpha*self.X_f.b.reshape(-1,1) - Lambda_X_f_w@h_w_til)

                # ensure containment in tightened state constraints [(4.25h) in the thesis]
                constraints.append(Lambda_xf@self.X_f.A == self.alpha*H_x)
                constraints.append(Lambda_xf@self.X_f.b.reshape(-1,1) <= h_x - Lambda_x[-n_x:]@np.kron(np.ones((N,1)), h_w_til))

                # ensure containment in tightened input constraints [(4.25i) in the thesis]
                constraints.append(Lambda_uf@self.X_f.A == self.alpha*H_u@self.K_f)
                constraints.append(Lambda_uf@self.X_f.b.reshape(-1,1) <= h_u - Lambda_u[-n_u:]@np.kron(np.ones((N,1)), h_w_til))

                # terminal constraint
                constraints.append(self.X_f.A@self.z[-n:] <= self.alpha*self.X_f.b.reshape(-1,1))
            else:
                """
                If the terminal set is not scaled, we revert to formulation (4.19) in the thesis. This contains [1] as
                a special case.

                [1] S. Chen, V. M. Preciado, M. Morari, and N. Matni, "Robust Model Predictive Control with Polytopic
                    Model Uncertainty through System Level Synthesis", Automatica, 2024.
                """

                # define dual variable
                Lambda_xf = cp.Variable((self.X_f.b.shape[0],N*n_w_til), nonneg=True)

                # terminal constraint [(4.19g) in the thesis]
                constraints.append(Lambda_xf@np.kron(np.eye(N),H_w_til) == self.X_f.A@self.Phi_e[-n:,:])
                constraints.append(self.X_f.A@self.z[-n:] <= self.X_f.b.reshape(-1,1) - Lambda_xf@np.kron(np.ones((N,1)),h_w_til))

        ###

        # define the CVX optimization problem object
        self.prob = cp.Problem(cp.Minimize(objective), constraints)

    def _define_output_mapping(self):
        return_dict = {
            'control': self.v,
            'state': self.z,
            'p': self.p,
            'Phi_e': self.Phi_e,
            'Phi_k': self.Phi_k,
            'Sigma': self.Sigma + self.Sigma_diag,
        }

        if self.scale_terminal_set:
            # if the terminal set is scaled, we also return the scaling variables ...
            return_dict['alpha'] = self.alpha
            return_dict['Gamma'] = self.Gamma
            # ... and the terminal filter variables
            return_dict['Xi'] = self.Xi

        return return_dict