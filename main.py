import Fractal2D
import numpy as np
from sympy import *


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
    return np.array([[3*x**2-3*y**2,-6*x*y],[6*x*y,3*x**2-2*y]])

colours = []
for fan in range(1,10000):
	colours.append([np.random.rand(),np.random.rand(),np.random.rand()])



fractal = Fractal2D.Fractal2D(F, J, maxIterations = 500, tolerance = 10e-4, simplified = True)

fractal.colours = colours
