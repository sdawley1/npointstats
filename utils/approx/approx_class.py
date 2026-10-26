#!/bin/bash
"""
Sam Dawley
05/2025
"""

from abc import ABC, abstractmethod


class Approx(ABC):
    """
    Base class for approximations
    """

    def __init__(self) -> None:
        """
        Initialize the approximation class
        """
        return

    def __repr__(self) -> str:
        """
        String representation of the class
        """
        return f"{self.__class__.__name__}()"

    @abstractmethod
    def apx_dcf(self, d: int, r: float, *args) -> float:
        if d == 1:
            return self.apx_dcf_1d(r, *args)
        elif d == 2:
            return self.apx_dcf_2d(r, *args)
        elif d == 3:
            return self.apx_dcf_3d(r, *args)
        else:
            return 1

    @abstractmethod
    def apx_dcf_1d(self, r: float) -> float:
        return

    @abstractmethod
    def apx_dcf_2d(self, r: float) -> float:
        return

    @abstractmethod
    def apx_dcf_3d(self, r: float) -> float:
        return


def main() -> None:
    """
    DOCSTRING
    """
    return


if __name__ == "__main__":
    main()
