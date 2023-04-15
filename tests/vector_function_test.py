import pytest
from expects import expect, equal
import numpy as np
from numpy.testing import assert_array_almost_equal

from autodiff_team29 import Node
from autodiff_team29 import VectorFunction
import autodiff_team29.elementaries as E


def test_vector_function():
    """
    Test VectorFunction correctly calculates new symbol, value, and jacobian.
    The function we are testing in this case is

    f([f1,f2]) = [ x1x2  + sin(x1)  ]
                 [ x1+x2 + sin(x1x2)]

    The jacobian of this function is as follows
    [ x2 + cos(x1)    ,              x1 ]
    [ 1 + x2cos(x1x2) , 1 + x1cos(x1x2) ]

    """
    x1 = Node("x1", np.pi, 1, seed_vector=[1, 0])
    x2 = Node("x2", np.pi / 2, 1, seed_vector=[0, 1])

    f1 = x1 * x2 + E.sin(x1)
    f2 = x1 + x2 + E.sin(x1 * x2)

    f = VectorFunction([f1, f2])

    expected_symbol = "['((x1*x2)+sin(x1))' '((x1+x2)+sin((x1*x2)))']"
    expected_value = np.array(
        [
            np.pi * np.pi / 2 + np.sin(np.pi),
            np.pi + np.pi / 2 + np.sin(np.pi * np.pi / 2),
        ]
    )
    expected_jacobian = np.array(
        [
            [np.pi / 2 + np.cos(np.pi), np.pi],
            [
                1 + np.pi / 2 * np.cos(np.pi * np.pi / 2),
                1 + np.pi * np.cos(np.pi * np.pi / 2),
            ],
        ]
    )

    expect(f.symbol).to(equal(expected_symbol))
    assert_array_almost_equal(f.value, expected_value)
    assert_array_almost_equal(f.jacobian, expected_jacobian)
