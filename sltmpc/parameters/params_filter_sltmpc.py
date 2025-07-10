from collections.abc import Callable
from dataclasses import dataclass, field
import numpy as np

from ampyc.typing import Noise
from ampyc.params import ParamsBase
from ampyc.noise import PolytopeNoise
from ampyc.utils import Polytope

from sltmpc.systems import FixedVertexUncertainty, Uncertainty


class FilterSLTMPCParams(ParamsBase):
    '''
    Default parameters for experiments with a filter-based SLTMPC controller.
    '''

    @dataclass
    class ctrl:
        name: str = 'Filter-based SLTMPC'
        N: int = 5
        Q: np.ndarray = 10 * np.eye(2)
        R: np.ndarray = 1 * np.eye(1)

    @dataclass
    class sys:
        # system dimensions
        n: int = 2
        m: int = 1

        # dynamics matrices
        A: np.ndarray = np.array(
            [
                [1.0, 0.15],
                [0.1, 1.0]
            ])
        B: np.ndarray = np.array([0.1, 1.1]).reshape(-1,1)

        # state constraints
        A_x: np.ndarray | None = np.array(
            [
                [1, 0], 
                [-1, 0],
                [0, 1],
                [0, -1]
            ])
        b_x: np.ndarray | None = np.array([8.0, 8.0, 8.0, 8.0]).reshape(-1,1)

        # input constraints
        A_u: np.ndarray | None = np.array([1, -1]).reshape(-1,1)
        b_u: np.ndarray | None = np.array([4.0, 4.0]).reshape(-1,1)

        # model uncertainty description
        eps_A: float = 0.1
        eps_B: float = 0.1

        Delta_A: list[np.ndarray] = field(default_factory= lambda: [np.zeros((1,))])
        Delta_B: list[np.ndarray] = field(default_factory= lambda: [np.zeros((1,))])

        # uncertainty generators
        Delta_A_gen: Uncertainty = None
        Delta_B_gen: Uncertainty = None
        
        # noise description
        sig_w: float = 0.1
        A_w: np.ndarray | None = np.array(
            [
                [1, 0], 
                [-1, 0],
                [0, 1],
                [0, -1]
            ])
        b_w: np.ndarray | None = np.array([sig_w, sig_w, sig_w, sig_w]).reshape(-1,1)

        # noise generator
        noise_generator: Noise = PolytopeNoise(Polytope(A_w, b_w))

        def __post_init__(self) -> None:
            '''
            Post-initialization: ensure that derived attributes, i.e., parameters that are computed from other static parameters,
            are set correctly.
            '''
            # model uncertainty description
            self.Delta_A = [
                np.array([[1, 0], [0, 0]]) * self.eps_A,
                np.array([[-1, 0], [0, 0]]) * self.eps_A,
            ]
            self.Delta_B= [
                np.array([[0], [1]]) * self.eps_B,
                np.array([[0], [-1]]) * self.eps_B,
            ]

            # uncertainty generators
            self.Delta_A_gen = FixedVertexUncertainty(self.Delta_A)
            self.Delta_B_gen = FixedVertexUncertainty(self.Delta_B)

            # disturbance polytope
            self.b_w = np.array([self.sig_w, self.sig_w, self.sig_w, self.sig_w]).reshape(-1,1)

            # noise generator
            self.noise_generator = PolytopeNoise(Polytope(self.A_w, self.b_w))
    
    @dataclass
    class sim:
        num_steps: int = 25
        num_traj: int = 20
        x_0: np.ndarray = np.array([-7.0, 0.0]).reshape(-1,1)

    @dataclass
    class plot:
        color: str = 'blue'
        alpha: float | Callable = 1.0
        linewidth: float = 1.0
