import numpy as np
from autodiff_team29.node import Node
from autodiff_team29 import VectorFunction

# Vector Case
x1 = Node("x1", np.pi / 2, 1, seed_vector=[1, 0, 0])
x2 = Node("x2", np.pi / 16, 1, seed_vector=[0, 1, 0])
x3 = Node("x3", np.e, 1, seed_vector=[0, 0, 1])

print("Node aspects:")
print(f"Symbols: {x1.symbol}, {x2.symbol}, {x3.symbol}")
print(f"Values: x1: {x1.value}, x2: {x2.value}, x3: {x3.value}")
print(f"Derivatives: x1: {x1.derivative}, x2: {x2.derivative}, x3: {x3.derivative}")

# >>> Node aspects:
# >>> Symbols: x1, x2, x3
# >>> Values: x1: 1.5707963267948966, x2: 0.19634954084936207, x3: 2.718281828459045
# >>> Derivatives: x1: [1 0 0], x2: [0 1 0], x3: [0 0 1]

# Defining a function `f`
f1 = x1 + x2 * x3

# Instantiating the VectorFunction
f = VectorFunction([f1])

# Inspect f --> new `Node` with consistent representation
print("Symbol:", f.symbol)
print("Value", f.value)
print("Jacobian", f.jacobian)

# >>> Symbol: ['((x2*x3)+x1)']
# >>> Value [2.10452972]
# >>> Jacobian [[1.         2.71828183 0.19634954]]
