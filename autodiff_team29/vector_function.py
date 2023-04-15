from typing import List

import numpy as np
from numpy.typing import NDArray

from autodiff_team29 import Node


class VectorFunction:
    def __init__(self, functions: List[Node]) -> None:
        """
        Computes forward mode automatic differentiation for provided functions

        Parameters
        ----------
        functions : Node or List[Node]
            functions that compose the vector function

        Raises:
        ------
        ValueError :
            Raise value error if functions is not List[Node]

        Example
        -------

        Consider the function f(x,y) = [x + y, x-2y].
        An instantiation would be
        >>> x = Node("x", 2, 1 ,seed_vector=[1,0])
        >>> y = Node("y", 3, 1, seed_vector=[0,1])
        >>> f = VectorFunction([x + y, x - 2 * y])

        """

        if isinstance(functions, list):
            assert all(isinstance(f, Node) for f in functions)
            self._functions = functions
        else:
            raise ValueError("functions argument must be a list of Nodes")

    @property
    def symbol(self) -> str:
        """
        Returns the symbolic representation of the vector function

        """
        return str(np.array([function.symbol for function in self._functions]))

    @property
    def value(self) -> NDArray[float]:
        """
        Returns the computed value of the vector function

        """
        return np.array([function.value for function in self._functions])

    @property
    def jacobian(self) -> NDArray[float]:
        """
        Returns the computed Jacobian of the vector function

        """
        return np.array([function.derivative for function in self._functions])
