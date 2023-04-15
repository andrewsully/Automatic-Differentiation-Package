import pytest
import numpy as np
from expects import expect, equal, raise_error, be_true, be_within

from autodiff_team29 import elementaries
from autodiff_team29.node import Node


class TestDomainRestrictions:
    def test_log_domain_raises_error_for_values_outside_domain(self):
        """
        Checks that a ValueError is raised for values outside the domain of logarithmic functions
        """
        node = Node("x", -2, 1)
        with pytest.raises(ValueError):
            expect(elementaries._check_log_domain_restrictions(node)).to(
                raise_error(ValueError)
            )

    def test_sqrt_domain_raises_error_for_values_outside_domain(self):
        """
        Checks that a ValueError is raised for values outside the domain of sqrt
        """
        negative_two = Node("x", -2, 1)
        with pytest.raises(ValueError):
            elementaries._check_sqrt_domain_restrictions(negative_two)

    def test_arccos_domain_raises_error_for_values_outside_domain(self):
        """
        Checks that a ValueError is raised for values outside the domain of arccos
        """
        negative_two = Node("x", -2, 1)
        positive_two = Node("x", 2, 1)
        with pytest.raises(ValueError):
            elementaries._check_arccos_domain_restrictions(negative_two)
            elementaries._check_arccos_domain_restrictions(positive_two)

    def test_arcsin_domain_raises_error_for_values_outside_domain(self):
        """
        Checks that a ValueError is raised for values outside the domain of arcsin
        """
        negative_two = Node("x", -2, 1)
        positive_two = Node("x", 2, 1)
        with pytest.raises(ValueError):
            elementaries._check_arcsin_domain_restrictions(negative_two)
            elementaries._check_arcsin_domain_restrictions(positive_two)


class TestMathFunctions:
    def test_sqrt(self):
        """
        Test square root function
        """
        # int case
        value = 9
        expect(elementaries.sqrt(value).symbol).to(equal("sqrt(9)"))
        expect(elementaries.sqrt(value).value).to(equal(3.0))
        expect(elementaries.sqrt(value).derivative).to(equal(0))

        # float case
        value = 9.0
        expect(elementaries.sqrt(value).symbol).to(equal("sqrt(9.0)"))
        expect(elementaries.sqrt(value).value).to(equal(3.0))
        expect(elementaries.sqrt(value).derivative).to(equal(0))

        # node case
        value = Node("x", 9, 1)
        expect(elementaries.sqrt(value).symbol).to(equal("sqrt(x)"))
        expect(elementaries.sqrt(value).value).to(equal(3.0))
        expect(elementaries.sqrt(value).derivative).to(equal(1 / 6))

    def test_ln(self):
        """
        Test ln function
        """
        # int case
        value = 10
        expect(elementaries.ln(value).symbol).to(equal("ln(10)"))
        expect(elementaries.ln(value).value).to(equal(np.log(10)))
        expect(elementaries.ln(value).derivative).to(equal(1 / 10))

        # float case
        value = 10.0
        expect(elementaries.ln(value).symbol).to(equal("ln(10.0)"))
        expect(elementaries.ln(value).value).to(equal(np.log(10.0)))
        expect(elementaries.ln(value).derivative).to(equal(1 / 10))

        # node case
        value = Node("x", 10, 1)
        expect(elementaries.ln(value).symbol).to(equal("ln(x)"))
        expect(elementaries.ln(value).value).to(equal(np.log(10)))
        expect(elementaries.ln(value).derivative).to(equal(1 / 10))

    def test_log(self):
        """
        Test log10 function
        """
        # int case
        value = 10
        base = 10
        expect(elementaries.log(value, base).symbol).to(equal("log10(10)"))
        expect(elementaries.log(value, base).value).to(equal(np.log10(10)))
        expect(elementaries.log(value, base).derivative).to(
            equal(1 / (10 * np.log(10)))
        )

        # float case
        value = 10.0
        base = 2
        expect(elementaries.log(value, base).symbol).to(equal("log2(10.0)"))
        expect(np.isclose(elementaries.log(value, base).value, np.log2(10.0))).to(
            equal(True)
        )
        expect(
            np.isclose(elementaries.log(value, base).derivative, 1 / (10 * np.log(2)))
        ).to(equal(True))

        # node case
        value = Node("x", 10, 1)
        base = 10
        expect(elementaries.log(value, base).symbol).to(equal("log10(x)"))
        expect(np.isclose(elementaries.log(value, base).value, np.log10(10))).to(
            equal(True)
        )
        expect(
            np.isclose(elementaries.log(value, base).derivative, 1 / (10 * np.log(10)))
        ).to(equal(True))

    def test_exp(self):
        """
        Test exp function
        """
        # int case
        value = 1
        expect(elementaries.exp(value).symbol).to(equal("exp(1)"))
        expect(elementaries.exp(value).value).to(equal(np.exp(1)))
        expect(elementaries.exp(value).derivative).to(equal(0))

        # float case
        value = 1.0
        expect(elementaries.exp(value).symbol).to(equal("exp(1.0)"))
        expect(elementaries.exp(value).value).to(equal(np.exp(1.0)))
        expect(elementaries.exp(value).derivative).to(equal(0))

        # node case
        value = Node("x", 1, 1)
        expect(elementaries.exp(value).symbol).to(equal("exp(x)"))
        expect(elementaries.exp(value).value).to(equal(np.exp(1)))
        expect(elementaries.exp(value).derivative).to(equal(np.e))

    def test_sin(self):
        """
        Test sin function
        """
        # int case
        value = 1
        expect(elementaries.sin(value).symbol).to(equal("sin(1)"))
        expect(elementaries.sin(value).value).to(equal(np.sin(1)))
        expect(elementaries.sin(value).derivative).to(equal(0))

        # float case
        value = 1.0
        expect(elementaries.sin(value).symbol).to(equal("sin(1.0)"))
        expect(elementaries.sin(value).value).to(equal(np.sin(1.0)))
        expect(elementaries.sin(value).derivative).to(equal(0))

        # node case
        value = Node("x", 1, 1)
        expect(elementaries.sin(value).symbol).to(equal("sin(x)"))
        expect(elementaries.sin(value).value).to(equal(np.sin(1)))
        expect(elementaries.sin(value).derivative).to(equal(np.cos(1)))

    def test_cos(self):
        """
        Test cos function
        """
        # int case
        value = 1
        expect(elementaries.cos(value).symbol).to(equal("cos(1)"))
        expect(elementaries.cos(value).value).to(equal(np.cos(1)))
        expect(elementaries.cos(value).derivative).to(equal(0))

        # float case
        value = 1.0
        expect(elementaries.cos(value).symbol).to(equal("cos(1.0)"))
        expect(elementaries.cos(value).value).to(equal(np.cos(1.0)))
        expect(elementaries.cos(value).derivative).to(equal(0))

        # node case
        value = Node("x", 1, 1)
        expect(elementaries.cos(value).symbol).to(equal("cos(x)"))
        expect(elementaries.cos(value).value).to(equal(np.cos(1)))
        expect(elementaries.cos(value).derivative).to(equal(-np.sin(1)))

    def test_tan(self):
        """
        Test tan function
        """
        # int case
        value = 1
        expect(elementaries.tan(value).symbol).to(equal("tan(1)"))
        expect(elementaries.tan(value).value).to(equal(np.tan(1)))
        expect(elementaries.tan(value).derivative).to(equal(0))

        # float case
        value = 1.0
        expect(elementaries.tan(value).symbol).to(equal("tan(1.0)"))
        expect(elementaries.tan(value).value).to(equal(np.tan(1.0)))
        expect(elementaries.tan(value).derivative).to(equal(0))

        # node case
        value = Node("x", 1, 1)
        expect(elementaries.tan(value).symbol).to(equal("tan(x)"))
        expect(elementaries.tan(value).value).to(equal(np.tan(1)))
        expect(elementaries.tan(value).derivative).to(equal(1 / (np.cos(1) ** 2)))

    def test_arcsin(self):
        """
        Test arcsin function
        """
        # int case
        value = 1 / 2
        expect(elementaries.arcsin(value).symbol).to(equal("arcsin(0.5)"))
        expect(elementaries.arcsin(value).value).to(equal(np.arcsin(0.5)))
        expect(elementaries.arcsin(value).derivative).to(equal(0))

        # float case
        value = 0.5
        expect(elementaries.arcsin(value).symbol).to(equal("arcsin(0.5)"))
        expect(elementaries.arcsin(value).value).to(equal(np.arcsin(0.5)))
        expect(elementaries.arcsin(value).derivative).to(equal(0))

        # node case
        value = Node("x", 1 / 2, 1)
        expect(elementaries.arcsin(value).symbol).to(equal("arcsin(x)"))
        expect(elementaries.arcsin(value).value).to(equal(np.arcsin(0.5)))
        expect(elementaries.arcsin(value).derivative).to(
            equal(1 / np.sqrt(1 - 0.5**2))
        )

    def test_arccos(self):
        """
        Test arcsin function
        """
        # int case
        value = 1 / 2
        expect(elementaries.arccos(value).symbol).to(equal("arccos(0.5)"))
        expect(elementaries.arccos(value).value).to(equal(np.arccos(0.5)))
        expect(elementaries.arccos(value).derivative).to(equal(0))

        # float case
        value = 0.5
        expect(elementaries.arccos(value).symbol).to(equal("arccos(0.5)"))
        expect(elementaries.arccos(value).value).to(equal(np.arccos(0.5)))
        expect(elementaries.arccos(value).derivative).to(equal(0))

        # node case
        value = Node("x", 1 / 2, 1)
        expect(elementaries.arccos(value).symbol).to(equal("arccos(x)"))
        expect(elementaries.arccos(value).value).to(equal(np.arccos(0.5)))
        expect(elementaries.arccos(value).derivative).to(
            equal(-1 / np.sqrt(1 - 0.5**2))
        )

    def test_arctan(self):
        """
        Test arcsin function
        """
        # int case
        value = 1 / 2
        expect(elementaries.arctan(value).symbol).to(equal("arctan(0.5)"))
        expect(elementaries.arctan(value).value).to(equal(np.arctan(0.5)))
        expect(elementaries.arctan(value).derivative).to(equal(0))

        # float case
        value = 0.5
        expect(elementaries.arctan(value).symbol).to(equal("arctan(0.5)"))
        expect(elementaries.arctan(value).value).to(equal(np.arctan(0.5)))
        expect(elementaries.arctan(value).derivative).to(equal(0))

        # node case
        value = Node("x", 1 / 2, 1)
        expect(elementaries.arctan(value).symbol).to(equal("arctan(x)"))
        expect(elementaries.arctan(value).value).to(equal(np.arctan(0.5)))
        expect(elementaries.arctan(value).derivative).to(equal(1 / (1 + 0.5**2)))
       
    def test_power(self):
        """
        Test power function
        """
        # int case
        value = 10
        base = 10
        expect(elementaries.power(value, base).symbol).to(equal("(10**10)"))
        expect(elementaries.power(value, base).value).to(equal(np.power(10,10)))
        expect(elementaries.power(value, base).derivative).to(equal(0))

        # float case
        value = 10.0
        base = 2
        expect(elementaries.power(value, base).symbol).to(equal("(10.0**2)"))
        expect(np.isclose(elementaries.power(value, base).value, np.power(10.0, 2))).to(equal(True))
        expect(elementaries.power(value, base).derivative).to(equal(0))

        # node case
        value = Node("x", 10, 1)
        base = 10
        expect(elementaries.power(value, base).symbol).to(equal("(x**10)"))
        expect(np.isclose(elementaries.power(value, base).value, np.power(10,10))).to(equal(True))
        expect(elementaries.power(value, base).derivative).to(equal(np.power(10,10)))

    def test_sinh(self):
        """
        Test sinh function
        """
        # int case
        value = 1
        expect(elementaries.sinh(value).symbol).to(equal("sinh(1)"))
        expect(elementaries.sinh(value).value).to(equal(np.sinh(1)))
        expect(elementaries.sinh(value).derivative).to(equal(0))

        # float case
        value = 1.0
        expect(elementaries.sinh(value).symbol).to(equal("sinh(1.0)"))
        expect(elementaries.sinh(value).value).to(equal(np.sinh(1.0)))
        expect(elementaries.sinh(value).derivative).to(equal(0))

        # node case
        value = Node("x", 1, 1)
        expect(elementaries.sinh(value).symbol).to(equal("sinh(x)"))
        expect(elementaries.sinh(value).value).to(equal(np.sinh(1)))
        expect(elementaries.sinh(value).derivative).to(equal(np.cosh(1)))

    def test_cosh(self):
        """
        Test cosh function
        """
        # int case
        value = 1
        expect(elementaries.cosh(value).symbol).to(equal("cosh(1)"))
        expect(elementaries.cosh(value).value).to(equal(np.cosh(1)))
        expect(elementaries.cosh(value).derivative).to(equal(0))

        # float case
        value = 1.0
        expect(elementaries.cosh(value).symbol).to(equal("cosh(1.0)"))
        expect(elementaries.cosh(value).value).to(equal(np.cosh(1.0)))
        expect(elementaries.cosh(value).derivative).to(equal(0))

        # node case
        value = Node("x", 1, 1)
        expect(elementaries.cosh(value).symbol).to(equal("cosh(x)"))
        expect(elementaries.cosh(value).value).to(equal(np.cosh(1)))
        expect(elementaries.cosh(value).derivative).to(equal(np.sinh(1)))
   
    def test_tanh(self):
        """
        Test tanh function
        """
        # int case
        value = 1
        expect(elementaries.tanh(value).symbol).to(equal("tanh(1)"))
        expect(elementaries.tanh(value).value).to(equal(np.tanh(1)))
        expect(elementaries.tanh(value).derivative).to(equal(0))

        # float case
        value = 1.0
        expect(elementaries.tanh(value).symbol).to(equal("tanh(1.0)"))
        expect(elementaries.tanh(value).value).to(equal(np.tanh(1.0)))
        expect(elementaries.tanh(value).derivative).to(equal(0))

        # node case
        value = Node("x", 1, 1)
        expect(elementaries.tanh(value).symbol).to(equal("tanh(x)"))
        expect(elementaries.tanh(value).value).to(equal(np.tanh(1)))
        expect(elementaries.tanh(value).derivative).to(equal(1 - np.tanh(1)**2))

    def test_logistic(self):
        """
        Test logistic function
        """
        # int case
        value = 1
        expect(elementaries.logistic(value).symbol).to(equal("logistic(1)"))
        expect(elementaries.logistic(value).value).to(equal(np.exp(-np.logaddexp(0, -1))))
        expect(elementaries.logistic(value).derivative).to(equal(0))

        # float case
        value = 1.0
        expect(elementaries.logistic(value).symbol).to(equal("logistic(1.0)"))
        expect(elementaries.logistic(value).value).to(equal(np.exp(-np.logaddexp(0, -1.0))))
        expect(elementaries.logistic(value).derivative).to(equal(0))

        # node case
        value = Node("x", 1, 1)
        expect(elementaries.logistic(value).symbol).to(equal("logistic(x)"))
        expect(elementaries.logistic(value).value).to(equal(np.exp(-np.logaddexp(0, -1))))
        sigmoid = np.exp(-np.logaddexp(0, -1))
        expect(elementaries.logistic(value).derivative).to(equal(sigmoid * (1 - sigmoid)))
