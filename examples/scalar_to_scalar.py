import numpy as np
from autodiff_team29.node import Node
from autodiff_team29.elementaries import sin


# Scalar case
x = Node(symbol="x", value=np.pi, derivative=1)

print("Node aspects:")
print(f"Symbol: {x.symbol}")
print(f"Value: {x.value}")
print(f"Derivative: {x.derivative}")

# >>> Node aspects:
# >>> Symbol: x
# >>> Value: 3.141592653589793
# >>> Derivative: 1

# Defining a function `f`
f = sin(x) + x


print("Representation:")
print(repr(f))

# >>> Representation:
# >>> Node((sin(x)+x),3.141592653589793,0.0)