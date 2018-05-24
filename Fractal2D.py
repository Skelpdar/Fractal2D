import numpy as np
from sympy import symbols, diff, Array

class Fractal2D:
	
	zeroes = []
	colours = []

	def __init__(self, function, derivative = None, maxIterations = 100, tolerance = 10E-4):
		self.function = function
		self.derivative = derivative
		self.maxIterations = maxIterations
		self.tolerance = tolerance

#How a functioned could be written in order for automated derivation to work.
#def f(x,y):
#   return Array([[x**3-3*x*y**2-1],[3*x**2*y-y**2]])	
	def automatedDeriv(self):
		x, y = symbols('x y')        
		a = f1_derivative_x = diff(self.function[0],x)  
		b = f1_derivative_y = diff(self.function[0],y)
		c = f2_derivative_x = diff(self.function[1],x) 
		d = f2_derivative_y = diff(self.function[1],y)		 
		return np.array([[a,b],[c,d]])
    
	def newton(self, p):
		#Should return the coordinate of the root and the number of iterations it took
		#return None on failure to find root?	
		for n in range (self.maxIterations):
			try:
				p = np.linalg.solve(self.derivative(p), -self.function(p)) + p
				if np.linalg.norm(self.function(p)) < self.tolerance:
					return (p,n)
			except:
				return None
		return None
	

	def findRootIndex(self, root):
		#Should return the index in 'zeroes' of the root, or an already present one within the given tolerance
		
		#Check if the difference to some existing root is smaller than the tolerance
		for index in range(len(self.zeroes)):
			if np.linalg.norm((self.zeroes[index] - root)) < self.tolerance:
				return index
		#If not, add a new root
		else:
			self.zeroes.append(root)
			return len(self.zeroes)-1	

	def getColor(self, p):
		root = self.newton(p)
		c = [[21/255,101/255,192/255],[253/255,229/255,1/255]]
		if root == None:
			return np.array([1,1,1])
		else:
			index = self.findRootIndex(root[0])
			return np.array(c[index]) * np.log(root[1])/np.log(self.maxIterations)  
	
	def plot(self, dim, N):
		im = np.zeros([N,N,3])

		x = np.linspace(-dim,dim,N)
		y = np.linspace(-dim,dim,N)

		for ny in range(N):
			for nx in range(N):
				im[ny][nx] = self.getColor(np.array([x[nx],y[ny]]))
		
		plt.imshow(im, origin = 'lower', extent = [-dim,dim,-dim,dim])
		
	
	def simplifiedNewton(self, p):
		#Newtons method but with numericalDerivativ
		pass

	def numericalDerivative(self, p, stepDistance):
		#The function needs to have two inputs for this to work.        
		h = stepDistance
		f1x_ad_h = self.function(p[0]+h,p[1])[0]
		f1x_sub_h = self.function(p[0]-h,p[1])[0]
		f1y_ad_h = self.function(p[0],p[1]+h)[0]
		f1y_sub_h = self.function(p[0],p[1]-h)[0]
		f1_derivative_x = (f1x_ad_h-f1x_sub_h)/(2*h) 
		f1_derivative_y = (f1y_ad_h-f1y_sub_h)/(2*h)   
		
		f2x_ad_h = self.function(p[0]+h,p[1])[1]
		f2x_sub_h = self.function(p[0]-h,p[1])[1]
		f2y_ad_h = self.function(p[0],p[1]+h)[1]
		f2y_sub_h = self.function(p[0],p[1]-h)[1]
		f2_derivative_x = (f2x_ad_h-f2x_sub_h)/(2*h) 
		f2_derivative_y = (f2y_ad_h-f2y_sub_h)/(2*h)
		return np.array([[f1_derivative_x,f1_derivative_y],[f2_derivative_x,f2_derivative_y]])
	


