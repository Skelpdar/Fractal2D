import Fractal2D
import numpy as np
#from sympy import *

np.seterr(all = 'raise')

#Defines the function
def F(p):
	x = p[0]
	y = p[1]
	return np.array([x**8 - 28*x**6*y**2 + 70*x**4*y**4 + 15*x**4 - 28*x**2*y**6 - 90*x**2*y**2 + y**8 + 15*y**4 - 16,
				  8*x**7*y - 56*x**5*y**3 +56*x**3*y**5 + 60*x**3*y - 8*x*y**7 - 60*x*y**3])

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
for c in range(1,10000):
	colours.append([np.random.rand(),np.random.rand(),np.random.rand()])

fractal = Fractal2D.Fractal2D(F, None, maxIterations = 500, tolerance = 10e-4, simplified = False)

fractal.colours = colours

fractal.colours[0] = [217/255,34/255,34/255]
fractal.colours[1] = [36/255,69/255,145/255]
fractal.colours[2] = [28/255,174/255,28/255]

p = np.array([1,1])
#print(fractal.derivative(p))

print(fractal.numericalDerivative(p))

fractal.plot(-2,2,-1.125,1.125,1280,720)
