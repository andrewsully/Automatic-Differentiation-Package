from autodiff_team29.node import Node
from autodiff_team29.elementaries import sin, cos, exp
from autodiff_team29 import VectorFunction


# 'Scalar to Vector' case
x = Node("x", value=2, derivative=1)


print("Node aspects:")
print(f"Symbol: {x.symbol}")
print(f"Value: {x.value}")
print(f"Derivative: {x.derivative}")

# >>> Node aspects:
# >>> Symbol: x
# >>> Value: 2
# >>> Derivative: 1


# Defining multiple functions: sin(x), cos(x) and exp(x)
f1, f2, f3 = sin(x), cos(x), exp(x)

# Instantiating the VectorFunction
f = VectorFunction([f1, f2, f3])

# Inspect f --> new `Node` with consistent representation
print("New Node representation:")
print("Symbol:", f.symbol)
print("Value", f.value)
print("Jacobian", f.jacobian)

# >>> New Node representation:
# >>> Symbol: ['sin(x)' 'cos(x)' 'exp(x)']
# >>> Value [ 0.90929743 -0.41614684  7.3890561 ]
# >>> Jacobian [-0.41614684 -0.90929743  7.3890561 ]
