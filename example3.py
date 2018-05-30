import Fractal2D
import numpy as np
from sympy import *

"""
In this example we run test considering plot2 function. 
"""

# Here you define your function. 

def F(p):
	x = p[0]
	y = p[1]
	return np.array([x**8-28*x**6*y**2+70*x**4*y**4+15*x**4-28*x**2*y**6-90*x**2*y**2+y**8+15*y**4-16, 
				  8*x**7*y-56*x**5*y**3+56*x**3*y**5+60*x**3*y-8*x*y**7-60*x*y**3])

"""
def F(p):
	x = p[0]
	y = p[1]
	return np.array([x**3-3*x*y**2-2*x-2, 3*x**2*y-y**3-2*y])
"""
"""
def F(p):
	x = p[0]
	y = p[1]
	return np.array([x**3-3*x*y**2-1, 3*x**2*y-y**3])
"""

# Here you can write the derivative by hand and use it in Fractal2D. If you don't
# use it, the numerical derivative wil be used. 
"""
def J(p):
    x = p[0]
    y = p[1]
    return np.array([[3*x**2-3*y**2,-6*x*y],[6*x*y,3*x**2-3*y**2]])
"""

"""
To test the simplifiedNewton method with plot2 you need to change self.newton
to self.simplifiedNewton in plot2 and use a selfmade derivative, J, in fractal. 
(SimplifiedNewton works directly with our main plot function)
"""

fractal = Fractal2D.Fractal2D(F, maxIterations = 100, tolerance = 10e-4, simplified = False)

fractal.plot2(720, 1280, -2, 2, -2, 2)
