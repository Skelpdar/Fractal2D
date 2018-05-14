import Fractal2D
import numpy as np

def F(p):
	x = p[0]
	y = p[1]
	return np.array([x**3-3*x*y**2-1,3*x**2*y-y**3])

def J(p):
    x = p[0]
    y = p[1]
    return np.array([[3*x**2-3*y**2,-6*x*y],[6*x*y,3*x**2-2*y]])

fractal = Fractal2D.Fractal2D(F, J, maxIterations = 10, tolerance = 10e-4)

print(fractal.newton(np.array([1,1])))
