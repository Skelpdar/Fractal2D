import numpy as np

class Fractal2D:
	
	zeroes = np.array([])

	def __init__(self, function, derivative = None, maxIterations = 100, tolerance = 10E-4):
		self.function = function
		self.derivative = derivative
		self.maxIterations = maxIterations
		self.tolerance = tolerance
	
	def newton(self, p):
		#Should return the coordinate of the root and the number of iterations it took
		#return None on failure to find root?	
		for n in range (self.maxIterations):
			p = np.linalg.solve(self.derivative(p), -self.function(p)) + p
			if np.linalg.norm(self.function(p)) < self.tolerance:
				return (p,n)
		return None
	

	def findZeroes(self, root, tolerance):
		#Should return the index in 'zeroes' of the root, or an already present one within the given tolerance
		pass

	def plot():
		pass
	
	def simplifiedNewton(self, p):
		#Newtons method but with numericalDerivativ
		pass

	def numericalDerivative(self, p, stepDistance):
		pass
	


