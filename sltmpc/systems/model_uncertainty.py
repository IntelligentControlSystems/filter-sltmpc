from abc import ABC, abstractmethod
from typing import TypeVar
import numpy as np


# Type variable for the uncertainty generators
Uncertainty = TypeVar('Uncertainty')

class UncertaintyBase(ABC):
    """Base class for random model uncertainty generators"""

    rng: np.random.Generator | None = None

    def generate(self, N: int | None = None) -> np.ndarray:
        return self._generate(N)

    @classmethod
    @abstractmethod
    def _generate(self, N: int | None = None) -> np.ndarray:
        """Uncertainty generation method to be implemented by the inherited class"""
        raise NotImplementedError

    def reset(self, *args, **kwargs):
        '''Calls the constructor again'''
        self.__init__(*args, **kwargs)


class ZeroUncertainty(UncertaintyBase):
    """Outputs zero uncertainty, i.e., no model uncertainty"""

    def __init__(self, dim: tuple[int]) -> None:
        self.dim = dim

    def _generate(self, N: int | None = None) -> np.ndarray:
        sample = np.zeros(self.dim) if N is None else [np.zeros(self.dim) for _ in range(N)]
        return sample


class VerticesUncertainty(UncertaintyBase):
    """Choses a random vertex of the uncertainty set as noise"""

    def __init__(self, D: list[np.ndarray], seed: int | None = None) -> None:
        assert seed is None or seed >= 0
        self.D = D
        self.rng = np.random.default_rng(seed)

    def _generate(self, N: int | None = None) -> np.ndarray:
        if N is None:
            idx = self.rng.choice(len(self.D))
            sample = self.D[idx]
        else:
            idx = self.rng.choice(len(self.D), N)
            sample = [self.D[i] for i in idx.tolist()]
        return sample


class FixedVertexUncertainty(UncertaintyBase):
    """Choses a random vertex of the uncertainty set, then keep it constant for all future calls"""

    def __init__(self, D: list[np.ndarray], seed: int | None = None) -> None:
        assert seed is None or seed >= 0
        self.D = D
        self.rng = np.random.default_rng(seed)

        idx = self.rng.choice(len(self.D))
        self.vertex = self.D[idx]

    def _generate(self, N: int | None = None) -> np.ndarray:
        if N is None:
            sample = self.vertex
        else:
            sample = [self.vertex] * N
        return sample


class UniformUncertainty(UncertaintyBase):
    """Samples a random model uncertainty within the uncertainty set, where vertices are weighted uniformly"""

    def __init__(self, D: list[np.ndarray], seed: int | None = None) -> None:
        assert seed is None or seed >= 0
        self.D = D
        self.dim = D[0].shape
        self.rng = np.random.default_rng(seed)

    def _generate(self, N: int | None = None) -> np.ndarray:
        """Based on implementation for randomPoint() in MPT"""
        if N is None:
            L = self.rng.uniform(size=(len(self.D),))
            L /= np.sum(L)
            sample = np.zeros(self.dim)
            for i in range(len(self.D)):
                sample += L[i].item() * self.D[i]
        else:
            L = self.rng.uniform(size=(N, len(self.D)))
            L /= np.sum(L, axis=1).reshape(-1, 1)
            sample = []
            for l in L:
                s = np.zeros(self.dim)
                for i in range(len(self.D)):
                    s += l[i].item() * self.D[i]
                sample.append(s)
        return sample
