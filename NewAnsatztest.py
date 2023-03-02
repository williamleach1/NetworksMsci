
from sympy import *

x = Symbol('x')

sigmoid = 1/(1+exp(-x))

sigmoid_deriv = diff(sigmoid, x)

print(sigmoid_deriv)

# n_l = z(-l)/(1+z(-l))^2

# Equation: sigmoid from 0 to L = N - 1 (given N) 
# 


