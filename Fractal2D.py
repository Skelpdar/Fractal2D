import numpy as np

class Fractal2D:
	
	zeroes = np.array([])

	def __init__(function, derivative = None, maxIterations = 100):
		self.function = function
		self.derivative = None
		self.maxIterations = maxIterations

	def newton(p):
		#Should return the coordinate of the root and the number of iterations it took
		#return None on failure to find root?	
		pass

	def findZeroes(root, tolerance):
		#Should return the index in 'zeroes' of the root, or an already present one within the given tolerance
		pass

	def plot():
		pass
	
	def simplifiedNewton(p):
		#Newtons method but with numericalDerivativ
		pass

	def numericalDerivative(p, stepDistance):
		pass
