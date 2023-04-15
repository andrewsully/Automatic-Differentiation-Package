import warnings

import pytest
from expects import expect, equal, be, be_none, be_true, be_empty, have_key
import numpy as np

from autodiff_team29.node import Node


class TestNodeRegistry:
    """
    Test that the node registry is properly adding and removing nodes.
    There are a lot of global variables in the class scope that require extensive testing.

    """

    def test_creating_new_symbol_appears_in_registry(self):
        """
        Verify that once a node is created, it can be found in the _NODE_REGISTRY
        by looking for its symbolic representation

        """
        x = Node("new_symbol", 100, 1)

        expect(Node._NODE_REGISTRY).to(have_key("new_symbol"))

    def test_clear_node_registry_removes_all_key_value_pairs(self):
        """
        Verify that Node.clear_registry() removes all keys from the _NODE_REGISTRY

        """
        Node._NODE_REGISTRY = {"key": "value"}
        Node.clear_node_registry()

        expect(Node._NODE_REGISTRY).to(be_empty)

    def test_clear_node_registry_resets_count_of_nodes(self):
        """
        Verify that Node.clear_registry() removes all keys from the _NODE_REGISTRY

        """
        Node._NODE_REGISTRY = {"key": "value"}
        Node.clear_node_registry()

        expect(Node.count_nodes_stored()).to(be(0))

    def test_precomputed_nodes_can_be_retrieved(self):
        """
        Verify that once a node is created, the exact same instance can be retrieved
        by looking for its symbolic representation

        """
        x = Node("a", 50, 1)
        retrieved_node = Node._get_existing_node("a")

        expect(retrieved_node).to(be(x))

    def test_incrementing_node_count_while_overwrite_mode_is_false(self):
        """
        If overwrite mode is off, then we expect that each unique node
        will increment the node count by one. Repeated symbols will not increment the count

        """
        # ensure overwrite mode is disabled
        Node.set_overwrite_mode(False)

        x = Node("first", 1, 1)
        expect(Node.count_nodes_stored()).to(equal(1))

        x = Node("second", 2, 1)
        expect(Node.count_nodes_stored()).to(equal(2))

        x = Node("third", 3, 1)
        expect(Node.count_nodes_stored()).to(equal(3))

        # repeat third symbol. count should not increment
        x = Node("third", 4, 1)
        expect(Node.count_nodes_stored()).to(equal(3))

    def test_incrementing_node_count_while_overwrite_mode_is_true(self):
        """ "
        If overwrite mode is on, then we expect the number of nodes stored to remain zero
        """

        # enabling overwrite mode
        Node.set_overwrite_mode(True)

        x = Node("first", 1, 1)
        expect(Node.count_nodes_stored()).to(equal(0))

        x = Node("second", 2, 1)
        expect(Node.count_nodes_stored()).to(equal(0))

        x = Node("third", 3, 1)
        expect(Node.count_nodes_stored()).to(equal(0))

        # repeat third symbol. count should remain zero
        x = Node("third", 4, 1)
        expect(Node.count_nodes_stored()).to(equal(0))



class TestNodeCreation:
    """
    Test instantiation of Node objects.
    Ensure that objects are instantiated correctly.

    """

    def test_simple_initialization(self, teardown_module):
        """
        Test simple initialization with only required positional arguments works correctly

        """
        node = Node("a", 10, 1)
        expect(node._symbol).to(equal("a"))
        expect(node._value).to(equal(10))
        expect(node._derivative).to(equal(1))

    def test_seed_vector_initialization(self):
        """
        Test initialization with optional kwargs works correctly

        """
        node = Node("b", 20, 1, seed_vector=[0, 1])
        expect(node._symbol).to(equal("b"))
        expect(node._value).to(equal(20))
        expect(all(node._derivative == np.array([0, 1]))).to(be_true)

    def test_value(self):
        """
        Testing the `value` @property method

        """
        node = Node("c", 10, 3)
        expect(node.value).to(equal(10))

        node = Node("d", 4, 2, seed_vector=[0, 1])
        expect(node.value).to(equal(4))

    def test_symbol(self):
        """
        Testing the `symbol` @property method

        """
        node = Node("c", 10, 3)
        expect(node.symbol).to(equal("c"))

        node = Node("d", 4, 2, seed_vector=[0, 1])
        expect(node.symbol).to(equal("d"))

    def test_derivative(self):
        """
        Testing the `derivative` @property method

        """
        node = Node("c", 10, 3)
        expect(node.derivative).to(equal(3))

        node = Node("d", 4, 2, seed_vector=[0, 1])
        expect(all(node.derivative == np.array([0, 2]))).to(be_true)

    @pytest.mark.parametrize("argument", [1, 1.0])
    def test_compatible_value_type_does_not_raise_error(self, argument):
        """
        Testing the method that checks if a datatype can be represented as a node.
        We need to test all cases that are included in the _COMPATIBLE_TYPES options.

        """
        expect(Node._check_foreign_value_type_compatibility(argument)).to(be_none)

    @pytest.mark.parametrize("argument", [1, 1.0])
    def test_compatible_derivative_type_does_not_raise_error(self, argument):
        """
        Testing the method that checks if a datatype can be represented as a node.
        We need to test all cases that are included in the _COMPATIBLE_TYPES options.

        """
        expect(Node._check_foreign_derivative_type_compatibility(argument)).to(be_none)

    @pytest.mark.parametrize("argument", ["1", [1, 2, 3], np.array([4, 5, 6])])
    def test_incompatible_value_type_does_raise_type_error(self, argument):
        """
        Check incompatible datatypes raise compatibility errors

        """
        with pytest.raises(TypeError):
            Node._check_foreign_value_type_compatibility(argument)

    @pytest.mark.parametrize("argument", ["1", [1, 2, 3]])
    def test_incompatible_derivative_type_does_raise_type_error(self, argument):
        """
        Check incompatible datatypes raise compatibility errors

        """
        with pytest.raises(TypeError):
            Node._check_foreign_derivative_type_compatibility(argument)

    @pytest.mark.parametrize(
        "value, expected_symbol", [(1, "1"), (2, "2"), (3.0, "3.0")]
    )
    def test_integers_and_floats_can_be_converted_to_node(self, value, expected_symbol):
        """
        Validates that integers and floats can be converted to nodes

        """
        node = Node._convert_numeric_type_to_node(value)

        expect(node.symbol).to(equal(expected_symbol))
        expect(node.value).to(equal(value))
        expect(node.derivative).to(equal(0))

    @pytest.mark.parametrize("value", ["1", [1, 2, 3], np.array([1, 2, 3])])
    def test_non_numeric_types_raises_type_error(self, value):
        """
        Verifies that attempting to convert a non-numeric type to a node raises TypeError

        """
        with pytest.raises(TypeError):
            Node._convert_numeric_type_to_node(value)


class TestNodeOperators:
    """
    Testing overridden __dunder__ operators (+ , - , / , *, etc.)

    """

    def test_add(self):
        """
        Function that tests correct output for __add__ method.

        """
        node1 = Node("v0", 100, 1)

        # int case
        other = 101
        new_node = node1 + other
        expect(new_node.symbol).to(equal("(101+v0)"))
        expect(new_node.value).to(equal(201))
        expect(new_node.derivative).to(equal(1))

        # float case
        other = 101.00
        new_node = node1 + other
        expect(new_node.symbol).to(equal("(101.0+v0)"))
        expect(new_node.value).to(equal(201.00))
        expect(new_node.derivative).to(equal(1))

        # node case
        other = Node("v1", 2, 1)
        new_node = node1 + other
        expect(new_node.symbol).to(equal("(v0+v1)"))
        expect(new_node.value).to(equal(102))
        expect(new_node.derivative).to(equal(2))

    def test_radd(self):
        """
        Function that tests correct output for __radd__ method.

        """
        node1 = Node("v0", 100, 1)

        # int case
        other = 101
        new_node = other + node1
        expect(new_node.symbol).to(equal("(101+v0)"))
        expect(new_node.value).to(equal(201))
        expect(new_node.derivative).to(equal(1))

        # float case
        other = 101.00
        new_node = other + node1
        expect(new_node.symbol).to(equal("(101.0+v0)"))
        expect(new_node.value).to(equal(201.00))
        expect(new_node.derivative).to(equal(1))

    def test_sub(self):
        """
        Function that tests correct output for __sub__ method

        """
        node1 = Node("v0", 100, 1)

        # int case
        other = 101
        new_node = node1 - other
        expect(new_node.symbol).to(equal("(v0-101)"))
        expect(new_node.value).to(equal(-1))
        expect(new_node.derivative).to(equal(1))

        # float case
        other = 101.00
        new_node = node1 - other
        expect(new_node.symbol).to(equal("(v0-101.0)"))
        expect(new_node.value).to(equal(-1.0))
        expect(new_node.derivative).to(equal(1))

        # node case
        other = Node("v1", 2, 1)
        new_node = node1 - other
        expect(new_node.symbol).to(equal("(v0-v1)"))
        expect(new_node.value).to(equal(98))
        expect(new_node.derivative).to(equal(0))

    def test_rsub(self):
        """
        Function that tests correct output for __rsub__ method

        """
        node1 = Node("v0", 100, 1)

        # int case
        other = 101
        new_node = other - node1
        expect(new_node.symbol).to(equal("(101-v0)"))
        expect(new_node.value).to(equal(1))
        expect(new_node.derivative).to(equal(-1))

        # float case
        other = 101.00
        new_node = other - node1
        expect(new_node.symbol).to(equal("(101.0-v0)"))
        expect(new_node.value).to(equal(1.0))
        expect(new_node.derivative).to(equal(-1))

    def test_mul(self):
        """
        Function that tests correct output for __mul__ method

        """
        node1 = Node("v0", 100, 1)

        # int case
        other = 101
        new_node = node1 * other
        expect(new_node.symbol).to(equal("(101*v0)"))
        expect(new_node.value).to(equal(10100))
        expect(new_node.derivative).to(equal(101))

        # float case
        other = 101.00
        new_node = node1 * other
        expect(new_node.symbol).to(equal("(101.0*v0)"))
        expect(new_node.value).to(equal(10100.0))
        expect(new_node.derivative).to(equal(101))

        # node case
        other = Node("v1", 101, 1)
        new_node = node1 * other
        expect(new_node.symbol).to(equal("(v0*v1)"))
        expect(new_node.value).to(equal(10100))
        expect(new_node.derivative).to(equal(201))

    def test_rmul(self):
        """
        Function that tests correct output for __rmul__ method

        """
        node1 = Node("v0", 100, 1)

        # int case
        other = 101
        new_node = other * node1
        expect(new_node.symbol).to(equal("(101*v0)"))
        expect(new_node.value).to(equal(10100))
        expect(new_node.derivative).to(equal(101))

        # float case
        other = 101.00
        new_node = other * node1
        expect(new_node.symbol).to(equal("(101.0*v0)"))
        expect(new_node.value).to(equal(10100.0))
        expect(new_node.derivative).to(equal(101))

    def test_truediv(self):
        """
        Function that tests correct output for __truediv__ method

        """
        node1 = Node("v0", 100, 1)

        # int case
        other = 101
        new_node = node1 / other
        expect(new_node.symbol).to(equal("(v0/101)"))
        expect(new_node.value).to(equal(100 / 101))
        expect(new_node.derivative).to(equal(1 / 101))

        # float case
        other = 101.0
        new_node = node1 / other
        expect(new_node.symbol).to(equal("(v0/101.0)"))
        expect(new_node.value).to(equal(100 / 101.0))
        expect(new_node.derivative).to(equal(1 / 101.0))

        # node case
        other = Node("v1", 101, 1)
        new_node = node1 / other
        expect(new_node.symbol).to(equal("(v0/v1)"))
        expect(new_node.value).to(equal(100 / 101))
        expect(new_node.derivative).to(equal(1 / 101**2))

    def test_rtruediv(self):
        """
        Function that tests correct output for __rtruediv__ method.

        """
        node1 = Node("v0", 100, 1)

        # int case
        other = 101
        new_node = other / node1
        expect(new_node.symbol).to(equal("(101/v0)"))
        expect(new_node.value).to(equal(101 / 100))
        expect(new_node.derivative).to(equal(-101 / 100**2))

        # float case
        other = 101.0
        new_node = other / node1
        expect(new_node.symbol).to(equal("(101.0/v0)"))
        expect(new_node.value).to(equal(101.0 / 100))
        expect(new_node.derivative).to(equal(-101.0 / 100**2))

    def test_neg(self):
        """
        Function that tests correct output for __neg__ method.

        """
        # int case
        node1 = Node("v2", 100, 1)

        new_node = Node.__neg__(node1)
        expect(new_node.symbol).to(equal("-v2"))
        expect(new_node.derivative).to(equal(-1))
        expect(new_node.value).to(equal(-100))

        # float case
        node1 = Node("v3", 100.00, 1)
        new_node = Node.__neg__(node1)
        expect(new_node.symbol).to(equal("-v3"))
        expect(new_node.derivative).to(equal(-1))
        expect(new_node.value).to(equal(-100.00))

    def test_pow(self):

        """
        Function that tests correct output for __pow__ method.

        """
        node1 = Node("v0", 3, 1)

        # int case
        other = 2
        new_node = node1**other
        expect(new_node.symbol).to(equal("(v0**2)"))
        expect(new_node.value).to(equal(9))
        expect(new_node.derivative).to(equal(6))

        # float case
        other = 2.0
        new_node = node1**other
        expect(new_node.symbol).to(equal("(v0**2.0)"))
        expect(new_node.value).to(equal(9))
        expect(new_node.derivative).to(equal(6))

        # node case
        other = Node("v1", 2, 1)
        new_node = node1**other
        expect(new_node.symbol).to(equal("(v0**v1)"))
        expect(new_node.value).to(equal(9))
        expect(np.isclose(new_node.derivative, 2 * 3**1 + 3**2 * np.log(3))).to(
            equal(True)
        )

    def test_rpow(self):

        """
        Function tests for __rpow__ method.

        """
        node1 = Node("v0", 3, 1)

        # int case
        other = 2
        new_node = other ** node1
        expect(new_node.symbol).to(equal("(2**v0)"))
        expect(new_node.value).to(equal(8))
        expect(new_node.derivative).to(equal(np.log(2) * 2**3))

        # float case
        other = 2.0
        new_node = other ** node1
        expect(new_node.symbol).to(equal("(2.0**v0)"))
        expect(new_node.value).to(equal(8))
        expect(new_node.derivative).to(equal(np.log(2) * 2**3))

    def test_str(self):
        """
        Function that tests correct output for __str__ method.

        """
        # int case
        node1 = Node("v1", 100, 1)
        expect(str(node1)).to(equal("v1"))

        # float case
        node2 = Node("v3", 100.00, 1)
        expect(str(node2)).to(equal("v3"))

    def test_repr(self):
        """
        Function that tests correct output for __repr__ method.

        """
        # int case
        node1 = Node("v1", 1000, 1)
        expect(repr(node1)).to(
            equal(f"Node({node1._symbol},{node1._value},{node1._derivative})")
        )

        # float case:
        node2 = Node("v2", 101.03, 2)
        expect(repr(node2)).to(
            equal(f"Node({node2._symbol},{node2._value},{node2._derivative})")
        )
