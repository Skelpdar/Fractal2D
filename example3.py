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
	return np.array([x**3-3*x*y**2-2*x-2, 3*x**2*y-y**3-2*y])


fractal = Fractal2D.Fractal2D(F, maxIterations = 100, tolerance = 10e-4, simplified = False)

fractal.plot2(-2, 2, -2, 2, 72, 128)
