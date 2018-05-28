import Fractal2D
import numpy as np
from sympy import *

np.seterr(all = 'raise')

#Defines the function
def F(p):
	x = p[0]
	y = p[1]
	return np.array([x**3-3*x*y**2-1,3*x**2*y-y**3])

"""
def F(x,y):
    x,y = symbols('x,y')
    return Array([[x**3-3*x*y**2-1],[3*x**2*y-y**2]])
"""


#Defines the Jacobian 'derivative'
def J(p):
    x = p[0]
    y = p[1]
    return np.array([[3*x**2-3*y**2,-6*x*y],[6*x*y,3*x**2-3*y**2]])

colours = []
for fan in range(1,10000):
	colours.append([np.random.rand(),np.random.rand(),np.random.rand()])



fractal = Fractal2D.Fractal2D(F, J, maxIterations = 500, tolerance = 10e-4, simplified = False)

fractal.colours = colours

fractal.colours[0] = [83/255,118/255,20/255]
fractal.colours[1] = [65/255,124/255,129/255]

p = np.array([1,1])
print(fractal.derivative(p))

print(fractal.numericalDerivative(p))

#fractal.plot(-2,2,-2,2,800,800)
