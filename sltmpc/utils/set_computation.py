import numpy as np

from ampyc.typing import System
from ampyc.utils import Polytope, _reduce


def _pre_set(Omega: Polytope, sys: System) -> Polytope:
    '''
    Compute the pre-set of the polytopic set Omega for any input in the
    input constraint set U.

    Args:
        Omega (Polytope): The polytopic set for which the pre-set is computed.
        sys (System): The dynamic system object.

    Returns:
        Polytope: The pre-set of Omega.
    '''

    # obtain system parameters
    U, A, B = (sys.U, sys.A, sys.B)
    n = A.shape[0]

    # build half-space representation of the pre-set
    F, f = (Omega.A, Omega.b)
    F_bar = []
    f_bar = []

    # handle dynamics
    F_bar.append(np.hstack([F@A, F@B]))
    f_bar.append(f)

    # add the input constraint set
    F_bar.append(np.hstack([np.zeros((U.A.shape[0], n)), U.A]))
    f_bar.append(U.b)

    # build lifted pre-set and remove redundancies
    pre_lifted = Polytope(np.vstack(F_bar), np.hstack(f_bar), lazy=True)
    pre_lifted = _reduce(pre_lifted)

    # project back to the original space
    dim = list(range(1, n + 1))
    pre = pre_lifted.project(dim)

    return _reduce(pre)

def _robust_pre_set(Omega: Polytope, sys: System) -> Polytope:
    '''
    Compute the robust pre-set of the polytopic set Omega for any input in the
    input constraint set U and disturbance in the disturbance set W.

    Args:
        Omega (Polytope): The polytopic set for which the robust pre-set is computed.
        sys (System): The dynamic system object.

    Returns:
        Polytope: The robust pre-set of Omega.
    '''

    # obtain system parameters
    U, A, B = (sys.U, sys.A, sys.B)
    n = A.shape[0]

    # robustify the pre-set
    Omega_r = Omega - sys.W

    # build half-space representation of the pre-set
    F, f = (Omega_r.A, Omega_r.b)
    F_bar = []
    f_bar = []

    # handle model uncertainty
    if hasattr(sys, 'Delta_A'):
        for dA in sys.Delta_A:
            if hasattr(sys, 'Delta_B'):
                for dB in sys.Delta_B:
                    F_bar.append(np.hstack([F@(A + dA), F@(B + dB)]))
                    f_bar.append(f)
            else:
                F_bar.append(np.hstack([F@(A + dA), F@B]))
                f_bar.append(f)
    else:
        if hasattr(sys, 'Delta_B'):
            for dB in sys.Delta_B:
                F_bar.append(np.hstack([F@A, F@(B + dB)]))
                f_bar.append(f)
        else:
            F_bar.append(np.hstack([F@A, F@B]))
            f_bar.append(f)

    # add the input constraint set
    F_bar.append(np.hstack([np.zeros((U.A.shape[0], n)), U.A]))
    f_bar.append(U.b)

    # build lifted pre-set and remove redundancies
    pre_lifted = Polytope(np.vstack(F_bar), np.hstack(f_bar), lazy=True)
    pre_lifted = _reduce(pre_lifted)

    # project back to the original space
    dim = list(range(1, n + 1))
    pre = pre_lifted.project(dim)

    return _reduce(pre)

def compute_mci(sys: System, max_iter: int = 20) -> Polytope:
    '''
    Compute the maximal control invariant (MCI) set for a dynamic system.

    Args:
        sys (System): The dynamic system object for which the MCI is computed.
        max_iter (int): Maximum number of iterations for convergence.
    
    Returns:
        Polytope: The maximal control invariant (MCI) set.
    '''
    iters = 0
    mci = Polytope(A=sys.X.A, b=sys.X.b, lazy=True)

    while iters < max_iter:
        iters += 1
        mci_pre = _pre_set(mci, sys)
        mci_next = mci.intersect(mci_pre)

        if mci_next.is_empty:
            print('MCI computation converged to an empty set after {0} iterations.'.format(iters))
            return Polytope()

        if mci == mci_next:
            print('MCI computation converged after {0} iterations.'.format(iters))
            break

        if iters == max_iter:
            print('MCI computation did not converge after {0} max iterations.'.format(iters))
            break

        mci = mci_next

    return mci

def compute_mrci(sys: System, max_iter: int = 20) -> Polytope:
    '''
    Compute the maximal robust control invariant (MRCI) for a dynamic system.

    Args:
        sys (System): The dynamic system object for which the MRCI is computed.
        max_iter (int): Maximum number of iterations for convergence.
    
    Returns:
        Polytope: The maximal robust control invariant (MRCI) set.
    '''
    iters = 0
    mrci = Polytope(A=sys.X.A, b=sys.X.b, lazy=True)

    while iters < max_iter:
        iters += 1
        mrci_pre = _robust_pre_set(mrci, sys)
        mrci_next = mrci_pre.intersect(mrci)

        if mrci_next.is_empty:
            print('MRCI computation converged to an empty set after {0} iterations.'.format(iters))
            return Polytope()

        if mrci == mrci_next:
            print('MRCI computation converged after {0} iterations.'.format(iters))
            break

        if iters == max_iter:
            print('MRCI computation did not converge after {0} max iterations.'.format(iters))
            break

        mrci = mrci_next

    return mrci

