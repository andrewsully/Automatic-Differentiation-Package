from typing import Union
import numpy as np
import math
from autodiff_team29 import Node


def _check_log_domain_restrictions(x: Node) -> None:
    """
    Checks if the value of a given input x is less than or equal to zero and therefore
    unable to be used as an input for a logrithmic function.

    Parameters
    ----------
    x: Node

    Returns
    -------
    Returns None
        if x > 0

    Raises
    ------
    ValueError
        if x <= 0.

    Examples
    --------
    >>> _check_log_domain_restrictions(Node("1",1,0))
    None
    >>> _check_log_domain_restrictions(Node("0",0,0))
    ValueError: Value 0 not valid for a logarithmic function
    >>> _check_log_domain_restrictions(Node("-1",-1,0))
    ValueError: Value '-1' not valid for a logarithmic functionNone

    """
    if x.value <= 0:
        raise ValueError(f"Value '{x.value} 'not valid for a logarithmic function")


def _check_sqrt_domain_restrictions(x: Node) -> None:
    """
    Checks if the value of a given input x is less zero and therefore
    unable to be used as an input for a square root function.

    Parameters
    ----------
    x : Node

    Returns
    -------
    None
        if x >= 0

    Raises
    ------
    ValueError
        if x < 0.

    Examples
    --------
    >>> _check_sqrt_domain_restrictions(Node("1",1,0))
    None
    >>> _check_sqrt_domain_restrictions(Node("0",0,0))
    None
    >>> _check_sqrt_domain_restrictions(Node("-1",-1,0))
    ValueError: Square roots of negative numbers not supported

    """
    if x.value < 0:
        raise ValueError("Square roots of negative numbers not supported")


def _check_tan_domain_restrictions(x: Node) -> None:
    """
    Checks if cosine of given value is zero and thus invalid for tangent.

    Parameters
    ----------
    x : Node

    Returns
    -------
    None
        if cos(x) != 0

    Raises
    ------
    ValueError
        if cos(x) == 0.

    Examples
    --------
    >>> _check_tan_domain_restrictions(Node("1",1,0))
    None
    >>> _check_tan_domain_restrictions(Node("0",0,0))
    None
    >>> _check_tan_domain_restrictions(Node("pi",np.pi/2,0))
    ValueError: Value, pi/2, not within domain of tan

    """
    if np.cos(x.value) == 0:
        raise ValueError(f"Value, {x.value}, not within domain of tan")


def _check_arccos_domain_restrictions(x: Node) -> None:
    """
    Checks if the value of a given input x is not -1 ≤ x ≤ 1 therefore
    unable to be used as an input for the arccos function.

    Parameters
    ----------
    x : Node

    Returns
    -------
    None
        if -1 ≤ x ≤ 1

    Raises
    ------
    ValueError
        if |x| > 1.

    Examples
    --------
    >>> _check_arccos_domain_restrictions(Node("1",1,0))
    None
    >>> _check_arccos_domain_restrictions(Node("0",0,0))
    None
    >>> _check_arccos_domain_restrictions(Node("-5",-1,0))
    ValueError: '-5' is not within the domain [-1,1] of f(x)=arccos(x)

    """
    if np.abs(x.value) > 1:
        raise ValueError(
            f"'{x.value}' is not within the domain [-1,1] of f(x)=arccos(x)"
        )


def _check_arcsin_domain_restrictions(x: Node) -> None:
    """
    Checks if the value of a given input x is not -1 ≤ x ≤ 1 and therefore
    unable to be used as an input for the arcsin function.

    Parameters
    ----------
    x : Node

    Returns
    -------
    None
        if -1 ≤ x ≤ 1

    Raises
    ------
    ValueError
        if x < 0.

    Examples
    --------
    >>> _check_arcsin_domain_restrictions(Node("1",1,0))
    None
    >>> _check_arcsin_domain_restrictions(Node("0",0,0))
    None
    >>> _check_arcsin_domain_restrictions(Node("-5",-1,0))
    ValueError: '-5' is not within the domain [-1,1] of f(x)=arcsin(x)
    """
    if np.abs(x.value) > 1:
        raise ValueError(f"{x.value} is not within the domain [-1,1] of f(x)=arcsin(x)")


def sqrt(x: Union[int, float, Node]) -> Node:
    """
    Takes in an instance of the Node class and returns a new node with its symbolic
    representation, forward trace, and tangent trace, which are based of the square
    root of the input node x.

    Parameters
    ----------
    x : Union[int, float, Node]

    Returns
    -------
    Node

    Examples
    --------
    >>> sqrt(Node("1",1,0))
    Node("sqrt(1)", 1, 0)
    >>> sqrt(Node("0",0,0))
    Node("sqrt(0)", 0, 0)
    >>> sqrt(Node("-1",-1,0))
    ValueError: Square roots of negative numbers not supported

    """
    symbolic_representation = "sqrt({})".format(str(x))
    if Node._check_node_exists(symbolic_representation):
        return Node._get_existing_node(symbolic_representation)

    x = Node._convert_numeric_type_to_node(x)

    _check_sqrt_domain_restrictions(x)

    forward_trace = np.sqrt(x.value)
    tangent_trace = x.derivative / (2 * np.sqrt(x.value))
    new_node = Node(symbolic_representation, forward_trace, tangent_trace)

    return new_node


def ln(x: Union[int, float, Node]) -> Node:
    """
    Takes in an instance of the Node class and returns a new node with its symbolic
    representation, forward trace, and tangent trace, which are based on the natural log
    of the input node x.

    Parameters
    ----------
    x : Union[int, float, Node]

    Returns
    -------
    Node

    Examples
    --------
    >>> ln(Node("1",1,0))
    Node("ln(1)", 0, 1)
    >>> ln(Node("0",0,0))
    ValueError: Value 0 not valid for a logarithmic function
    >>> ln(-1)
    ValueError: Value '-1' not valid for a logarithmic functionNone

    """
    symbolic_representation = "ln({})".format(str(x))
    if Node._check_node_exists(symbolic_representation):
        return Node._get_existing_node(symbolic_representation)

    x = Node._convert_numeric_type_to_node(x)

    _check_log_domain_restrictions(x)

    forward_trace = np.log(x.value)
    tangent_trace = 1 / x.value
    new_node = Node(symbolic_representation, forward_trace, tangent_trace)

    return new_node


def log(x: Union[int, float, Node], base: Union[int, float, Node] = np.e) -> Node:
    """
    Takes in an instance of the Node class and returns a new node with its symbolic
    representation, forward trace, and tangent trace, which are based on the logarithm
    of the input node x and the provided base.

    Parameters
    ----------
    x : Union[int, float, Node]

    base : Union[int, float]
        The desired base of the logorithm. Must be an integer or float greater than 1.

    Returns
    -------
    Node

    Examples
    --------
    >>> log(Node("1",1,0), 10)
    Node("log10(1)", 0, 0.4343)
    >>> log(Node("1",1,0), 2)
    Node("log2(1)", 0, 1.4427)
    >>> log(Node("0",0,0))
    ValueError: Value 0 not valid for a logarithmic function
    >>> log(Node("-1",-1,0))
    ValueError: Value -1 not valid for a logarithmic function

    """
    if not base > 1:
        raise ValueError("Base must be greater than 1")

    symbolic_representation = f"log{str(base)}({str(x)})"
    if Node._check_node_exists(symbolic_representation):
        return Node._get_existing_node(symbolic_representation)

    x = Node._convert_numeric_type_to_node(x)

    _check_log_domain_restrictions(x)

    forward_trace = math.log(x.value, base)
    tangent_trace = 1 / (x.value * np.log(base))
    new_node = Node(symbolic_representation, forward_trace, tangent_trace)

    return new_node


def exp(x: Union[int, float, Node]) -> Node:
    """
    Takes in an instance of the Node class and returns a new node with its symbolic
    representation, forward trace, and tangent trace, which are based on the exponential
    value of the input node x.

    Parameters
    ----------
    x : Union[int, float, Node]

    Returns
    -------
    Node

    Examples
    --------
    >>> exp(Node("1",1,0))
    Node("exp(1)", 2.7183, 0)
    >>> exp(Node("0",0,0))
    Node("exp(0)", 1, 0)
    >>> exp(Node("-1",-1,0))
    Node("exp(-1)", 0.3679, 0)

    """
    symbolic_representation = "exp({})".format(str(x))
    if Node._check_node_exists(symbolic_representation):
        return Node._get_existing_node(symbolic_representation)

    x = Node._convert_numeric_type_to_node(x)

    forward_trace = np.exp(x.value)
    tangent_trace = x.derivative * forward_trace
    new_node = Node(symbolic_representation, forward_trace, tangent_trace)

    return new_node


def sin(x: Union[int, float, Node]) -> Node:
    """
    Takes in an instance of the Node class and returns a new node with its symbolic
    representation, forward trace, and tangent trace, which are based on the sine
    of the input node x.

    Parameters
    ----------
    x : Union[int, float, Node]

    Returns
    -------
    Node

    Examples
    --------
    >>> sin(Node("1",1,0))
    Node("sin(1)", 0.8415, 0)
    >>> sin(Node("0",0,0))
    Node("sin(0)", 0, 0)
    >>> sin(Node("-1",-1,0))
    Node("sin(-1)", -0.8415, 0)

    """
    symbolic_representation = "sin({})".format(str(x))
    if Node._check_node_exists(symbolic_representation):
        return Node._get_existing_node(symbolic_representation)

    x = Node._convert_numeric_type_to_node(x)

    forward_trace = np.sin(x.value)
    tangent_trace = np.cos(x.value) * x.derivative
    new_node = Node(symbolic_representation, forward_trace, tangent_trace)

    return new_node


def cos(x: Union[int, float, Node]) -> Node:
    """
    Takes in an instance of the Node class and returns a new node with its symbolic
    representation, forward trace, and tangent trace, which are based on the cosine
    of the input node x.

    Parameters
    ----------
    x : Union[int, float, Node]

    Returns
    -------
    Node

    Examples
    --------
    >>> cos(Node("1",1,0))
    Node("cos(1)", 0.5403, 0)
    >>> cos(Node("0",1,0))
    Node("cos(0)", 1, 0)
    >>> cos(Node("-1",-1,0))
    Node("cos(-1)", -0.5403, 0)

    """
    symbolic_representation = "cos({})".format(str(x))
    if Node._check_node_exists(symbolic_representation):
        return Node._get_existing_node(symbolic_representation)

    x = Node._convert_numeric_type_to_node(x)

    forward_trace = np.cos(x.value)
    tangent_trace = -np.sin(x.value) * x.derivative
    new_node = Node(symbolic_representation, forward_trace, tangent_trace)

    return new_node


def tan(x: Union[int, float, Node]) -> Node:
    """
    Takes in an instance of the Node class and returns a new node with its symbolic
    representation, forward trace, and tangent trace, which are based on the tangent
    of the input node x.

    Parameters
    ----------
    x : Union[int, float, Node]

    Returns
    -------
    Node

    Examples
    --------
    >>> tan(Node("1",1,0))
    Node("tan(1)", 1.557, 0)
    >>> tan(Node("0",0,0))
    Node("tan(0)", 0, 0)
    >>> tan(Node("-1",-1,0))
    Node("tan(-1)", -1.557, 0)

    """
    symbolic_representation = "tan({})".format(str(x))
    if Node._check_node_exists(symbolic_representation):
        return Node._get_existing_node(symbolic_representation)

    x = Node._convert_numeric_type_to_node(x)

    _check_tan_domain_restrictions(x)

    forward_trace = np.tan(x.value)
    tangent_trace = x.derivative / (np.cos(x.value) ** 2)
    new_node = Node(symbolic_representation, forward_trace, tangent_trace)

    return new_node


def arcsin(x: Union[int, float, Node]) -> Node:
    """
    Takes in an instance of the Node class and returns a new node with its symbolic
    representation, forward trace, and tangent trace, which are based on the arcsin
    of the input node x.

    Parameters
    ----------
    x : Union[int, float, Node]

    Returns
    -------
    Node

    Examples
    --------
    >>> arcsin(Node("1",1,0))
    Node("arcsin(1)", 1.5708, 0)
    >>> arcsin(Node("0",0,0))
    Node("arcsin(0)", 0, 0)
    >>> arcsin(Node("-1",-1,0))
    Node("arcsin(-1)", -1.5708, 0)

    """
    symbolic_representation = "arcsin({})".format(str(x))
    if Node._check_node_exists(symbolic_representation):
        return Node._get_existing_node(symbolic_representation)

    x = Node._convert_numeric_type_to_node(x)

    _check_arcsin_domain_restrictions(x)

    forward_trace = np.arcsin(x.value)
    tangent_trace = x.derivative / np.sqrt(1 - x.value ** 2)
    new_node = Node(symbolic_representation, forward_trace, tangent_trace)

    return new_node


def arccos(x: Union[int, float, Node]) -> Node:
    """
    Takes in an instance of the Node class and returns a new node with its symbolic
    representation, forward trace, and tangent trace, which are based on the arccos
    of the input node x.

    Parameters
    ----------
    x : Union[int, float, Node]

    Returns
    -------
    Node

    Examples
    --------
    >>> arccos(Node("1",1,0))
    Node("arccos(1)", 3.1416, 0)
    >>> arccos(Node("0",0,0))
    Node("arccos(0)", 1.5708, 0)
    >>> arccos(Node("-1",-1,0))
    Node("arccos(-1)", -3.1416, 0)

    """
    symbolic_representation = "arccos({})".format(str(x))
    if Node._check_node_exists(symbolic_representation):
        return Node._get_existing_node(symbolic_representation)

    x = Node._convert_numeric_type_to_node(x)

    _check_arccos_domain_restrictions(x)

    forward_trace = np.arccos(x.value)
    tangent_trace = -x.derivative / np.sqrt(1 - x.value ** 2)
    new_node = Node(symbolic_representation, forward_trace, tangent_trace)

    return new_node


def arctan(x: Union[int, float, Node]) -> Node:
    """
    Takes in an instance of the Node class and returns a new node with its symbolic
    representation, forward trace, and tangent trace, which are based on the arctan
    of the input node x.

    Parameters
    ----------
    x : Union[int, float, Node]

    Returns
    -------
    Node

    Examples
    --------
    >>> arctan(Node("1",1,0))
    Node("arctan(1)", 0.7854, 0)
    >>> arctan(Node("0",0,0))
    Node("arctan(0)", 0, 0)
    >>> arctan(Node("-1",-1,0))
    Node("arctan(-1)", -0.7854, 0)

    """
    symbolic_representation = "arctan({})".format(str(x))
    if Node._check_node_exists(symbolic_representation):
        return Node._get_existing_node(symbolic_representation)

    x = Node._convert_numeric_type_to_node(x)

    forward_trace = np.arctan(x.value)
    tangent_trace = x.derivative / (1 + x.value ** 2)
    new_node = Node(
        symbolic_representation,
        forward_trace,
        tangent_trace,
    )

    return new_node


def power(base: Union[int, float, Node], exponent: Union[int, float, Node]) -> Node:
    """
    Takes in an instance of the Node class and returns a new node with its symbolic
    representation, forward trace, and tangent trace, which are based on the power
    of the input node x.

    Parameters
    ----------
    base : Union[int, float, Node]
    exponent : Union[int, float, Node]

    Returns
    -------
    Node

    Examples
    --------
    >>> power(3,2)
    Node("3**2", 9, 0)

    """
    symbolic_representation = f"({base}**{exponent})"
    if Node._check_node_exists(symbolic_representation):
        return Node._get_existing_node(symbolic_representation)

    base = Node._convert_numeric_type_to_node(base)

    return base ** exponent


def sinh(x: Union[int, float, Node]) -> Node:
    """
    Takes in an instance of the Node class and returns a new node with its symbolic
    representation, forward trace, and tangent trace, which are based on the sinh
    of the input node x.

    Parameters
    ----------
    x : Union[int, float, Node]

    Returns
    -------
    Node


    Examples
    --------
    >>> sinh(1)
    Node("sinh(1)", 1.1752011936438014, 0)

    """
    symbolic_representation = f"sinh({x})"
    if Node._check_node_exists(symbolic_representation):
        return Node._get_existing_node(symbolic_representation)

    x = Node._convert_numeric_type_to_node(x)

    forward_trace = np.sinh(x.value)
    tangent_trace = np.cosh(x.value) * x.derivative
    new_node = Node(symbolic_representation, forward_trace, tangent_trace)

    return new_node


def cosh(x: Union[int, float, Node]) -> Node:
    """
    Takes in an instance of the Node class and returns a new node with its symbolic
    representation, forward trace, and tangent trace, which are based on the sinh
    of the input node x.

    Parameters
    ----------
    x : Union[int, float, Node]

    Returns
    -------
    Node

    Examples
    --------
    >>> cosh(1)
    Node("cosh(1)", 1.5430806348152437, 0)

    """
    symbolic_representation = f"cosh({x})"
    if Node._check_node_exists(symbolic_representation):
        return Node._get_existing_node(symbolic_representation)

    x = Node._convert_numeric_type_to_node(x)

    forward_trace = np.cosh(x.value)
    tangent_trace = np.sinh(x.value) * x.derivative
    new_node = Node(symbolic_representation, forward_trace, tangent_trace)

    return new_node


def tanh(x: Union[int, float, Node]) -> Node:
    """
    Takes in an instance of the Node class and returns a new node with its symbolic
    representation, forward trace, and tangent trace, which are based on the
    of the input node x.

    Parameters
    ----------
    x : Union[int, float, Node]

    Returns
    -------
    Node


    Examples
    --------
    >>> tanh(1)
    Node("tanh(1)", 0.76159415595, 0)

    """
    symbolic_representation = f"tanh({x})"
    if Node._check_node_exists(symbolic_representation):
        return Node._get_existing_node(symbolic_representation)

    x = Node._convert_numeric_type_to_node(x)

    forward_trace = np.tanh(x.value)
    tangent_trace = (1 - np.tanh(x.value) ** 2) * x.derivative
    new_node = Node(symbolic_representation, forward_trace, tangent_trace)

    return new_node


def logistic(x: Union[int, float, Node]) -> Node:
    """
    Takes in an instance of the Node class and returns a new node with its symbolic
    representation, forward trace, and tangent trace, which are based on the
    of the input node x.

    Parameters
    ----------
    x : Union[int, float, Node]

    Returns
    -------
    Node

    Examples
    --------
    >>> logistic(1)
    Node("logistic(1)", 1.1752011936438014, 0)

    """
    symbolic_representation = f"logistic({x})"
    if Node._check_node_exists(symbolic_representation):
        return Node._get_existing_node(symbolic_representation)

    x = Node._convert_numeric_type_to_node(x)

    forward_trace = np.exp(-np.logaddexp(0, -x.value))
    tangent_trace = (
            (np.exp(-np.logaddexp(0, -x.value)))
            * (1 - np.exp(-np.logaddexp(0, -x.value)))
            * x.derivative
    )
    new_node = Node(symbolic_representation, forward_trace, tangent_trace)

    return new_node
