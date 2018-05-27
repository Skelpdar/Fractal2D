import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, Array

class Fractal2D:
	
	#Contains all roots found so far
	zeroes = []
	
	#Determines which colour each root should have
	colours = []

	def __init__(self, function, derivative = None, maxIterations = 100, tolerance = 10E-4,
			  simplified = False):
		self.function = function
		self.derivative = derivative
		self.maxIterations = maxIterations
		self.tolerance = tolerance
		self.simplified = simplified

	#Erik C		
	def automatedDeriv(self):
		"""
		With the class initialized with a 2D Array-function,this method
		returns its jacobian matrix.
		"""
		x, y = symbols('x y')        
		a = diff(self.function(x,y)[0],x) #f1_derivative_x  
		b = diff(self.function(x,y)[0],y) #f1_derivative_y
		c = diff(self.function(x,y)[1],x) #f2_derivative_x 
		d = diff(self.function(x,y)[1],y) #f2_derivative_y		 
		return np.array([[a,b],[c,d]])
    
	#Erik W
	def newton(self, p):
		"""
		Takes a 2D-vector p and finds a root using Newton's method.
		Returns a 2D-vector on convergence, and None on divergence.
		"""
		if self.derivative == None:
			self.derivative = self.numericalDerivative            
		for n in range (1, self.maxIterations):
			try:
				p = np.linalg.solve(self.derivative(p), -self.function(p)) + p
				if np.linalg.norm(self.function(p)) < self.tolerance:
					return (p,n)
			except:
				return None
		return None
	
	#Erik W
	def findRootIndex(self, root):
		"""
		Returns the index in zeroes of the point root, if it hasn't been found yet
		add root to zeroes.
		"""
		
		#Check if the difference to some existing root is smaller than the tolerance.
		for index in range(len(self.zeroes)):
			if np.linalg.norm((self.zeroes[index] - root)) < self.tolerance:
				return index
		#If not, add a new root.
		else:
			self.zeroes.append(root)
			return len(self.zeroes)-1	
	
	#Viktor, Erik W
	def getColor(self, p):
			if not self.simplified:
				root = self.newton(p)
			else:
				root = self.simplifiedNewton(p)
			
			#c = [[21/255,101/255,192/255],[253/255,229/255,1/255]]
			if root == None:
				return np.array([1,1,1])
			else:
				index = self.findRootIndex(root[0])
				return np.array(self.colours[index]) * (1 - np.log(root[1])/np.log(self.maxIterations) * 0.95) 
	
	#Viktor, Erik W
	def plot(self, a,b,c,d,N,M):
		im = np.zeros([M,N,3])

		x = np.linspace(a,b,N)
		y = np.linspace(c,d,M)
		
		for ny in range(M):
			#print progress
			if ny % 10 == 0:
				print(ny)
		
			for nx in range(N):
				im[ny][nx] = self.getColor(np.array([x[nx],y[ny]]))
		
		#plt.imshow(im, origin = 'lower', extent = [a,b,c,d])
		plt.imsave('fractal.png',im, origin = 'lower')
	
	#Jonathan	
	def plot2(self, N, a, b, c, d):
		xvalues = np.linspace(a, b, N)
		yvalues = np.linspace(c, d, N)
		G = np.meshgrid(xvalues, yvalues)
		#A = []
		for n in range(N):
			#row = []
			for i in range(N):
				root = self.newton(np.array([G[0][n][i], G[1][n][i]]))
				if root == None:
					#row.append(5)
					G[0][n][i] = 5
				else:
					#row.append(self.findRootIndex(root[0]))
					G[0][n][i] = self.findRootIndex(root[0])
		#vstack(A)
		#plt.pcolor(A)
		plt.pcolor(G[0])
	
	#Harald, Erik W
	def simplifiedNewton(self, p):
		#Newtons method but with numericalDerivativ
		Jackinv = np.linalg.inv(self.derivative(p))
		for n in range(1,self.maxIterations):
			try:
				p = np.matmul(-Jackinv,self.function(p)) + p
				if np.linalg.norm(self.function(p)) < self.tolerance:
					return (p,n)
			except:
				return None
	
	#Erik C
	def numericalDerivative(self, p, stepDistance=0.001):
		"""
		Four numerical partial derivatives are evaluated at
		the given vector p. These are returned in a jacobian
		matrix of the function initialized in this class.
		"""        
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
