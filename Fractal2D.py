"""
vi3022jo-s@student.lu.se #Viktor Hrannar Jónsson
al5878la-s@student.lu.se #Alfred Langerbeck
er5872wa-s@student.lu.se #Erik Wallin
jo4450pe-s@student.lu.se #Jonathan Petersson
ha3247oh-s@student.lu.se #Harald Öhrn
er5046cl-s@student.lu.se #Erik Clarkson
"""

import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, Array

class Fractal2D:
		
			#Contains all roots found so far
			zeroes = []
			
			#Determines which colour each root should have
			colours = []
		
			#Viktor, Erik W, Alfred, Erik C, Jonathan, Harald, Victor
			def __init__(self, function, derivative = None, maxIterations = 100, tolerance = 10E-4, simplified = False, stepDistance = 0.001):
					if maxIterations <= 0:
						raise Exception('Max iterations should be over 0')
					self.function = function
					self.derivative = derivative
					self.maxIterations = maxIterations
					self.tolerance = tolerance
					self.simplified = simplified
					self.stepDistance = stepDistance
			#Erik C		
			def symbolicDerivative(self):
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
		    
		
			#Erik W, Jonathan, Erik C
			def newton(self, p):
				"""
				Takes a 2D-vector p and finds a root using Newton's method.
				Returns a 2D-vector on convergence, and None on divergence.
				"""         
				for n in range (1, self.maxIterations):
					try:
						if self.derivative == None:
							p = np.linalg.solve(self.numericalDerivative(p), -self.function(p)) + p
						else:
							p = np.linalg.solve(self.derivative(p), -self.function(p)) + p
						
						if np.linalg.norm(self.function(p)) < self.tolerance:
							return (p,n)
					except ValueError:
						print('A valueError occured in the newton method')
					except TypeError:
						print('A type error occured in the newton method, p should be a 2D array')
					except:
						return None
				return None
			
			#Erik W, Harald
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
				"""
				Fetches the nearest root using the newtons method for a specific 
				coordinate and assigns it a color using the findRootIndex. Then 
				shades the color depending on the number of iteration sit took to
				find the root.
				"""
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
			
			#Viktor, Erik W, Alfred
			def plot(self, a,b,c,d,N,M,Filename = 'fractal'):
				"""
				Creates a three layered matrix and goes through each element, using getColor to color that specific element.
				"""
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
				plt.imsave('{}.png'.format(Filename),im, origin = 'lower')
			
			#Jonathan	, Alfred
			def plot2(self, N, M, a, b, c, d,Filename = 'fractal2'):
				"""
				This is a plot function containing the commands meshgrid and pcolor 
				which was suggested i task 4. Later we improved this and did not use 
				either of the commands. What u see in # is an example of creating a 
				totally new matrix. What we are doing now is changing the elements 
				in one of the matrices in the meshgrid. 
				"""
				xvalues = np.linspace(a, b, M)
				yvalues = np.linspace(c, d, N)
				G = np.meshgrid(xvalues, yvalues)
				#A = []
				for n in range(N):
					#row = []
					for i in range(M):
						root = self.newton(np.array([G[0][n][i], G[1][n][i]]))
						if root == None:
							#row.append(5)
							G[0][n][i] = 5
						else:
							#row.append(self.findRootIndex(root[0]))
							G[0][n][i] = (1 + self.findRootIndex(root[0])) * (1 - np.log(root[1])/np.log(self.maxIterations) * 0.95)
				#vstack(A)
				#plt.pcolor(A)
				plt.pcolor(G[0])
				plt.savefig('{}.png'.format(Filename), dpi = 1200)
			
			#Harald, Erik W
			def simplifiedNewton(self, p):
				"""
				Newtons method but with numericalDerivativ
				"""
				try:
					Jackinv = np.linalg.inv(self.derivative(p))
					for n in range(1,self.maxIterations):
						try:
							p = np.matmul(-Jackinv,self.function(p)) + p
							if np.linalg.norm(self.function(p)) < self.tolerance:
								return (p,n)
						except:
							return None
				except np.LinAlgError:
					print('p is a singular matrix, SimplfiedNewton will not work!')
					
			#Erik C, Erik W
			def numericalDerivative(self, p):
				"""
				Four numerical partial derivatives are evaluated at
				the given vector p. These are returned in a jacobian
				matrix of the function initialized in this class.
				"""        
				h = self.stepDistance
				f1x_ad_h = self.function(p + np.array([h,0]))[0]
				f1x_sub_h = self.function(p + np.array([-h,0]))[0]
				f1y_ad_h = self.function(p + np.array([0,h]))[0]
				f1y_sub_h = self.function(p + np.array([0,-h]))[0]
				f1_derivative_x = (f1x_ad_h-f1x_sub_h)/(2*h) 
				f1_derivative_y = (f1y_ad_h-f1y_sub_h)/(2*h)   
				
				f2x_ad_h = self.function(p + np.array([h,0]))[1]
				f2x_sub_h = self.function(p + np.array([-h,0]))[1]
				f2y_ad_h = self.function(p + np.array([0,h]))[1]
				f2y_sub_h = self.function(p + np.array([0,-h]))[1]
				f2_derivative_x = (f2x_ad_h-f2x_sub_h)/(2*h) 
				f2_derivative_y = (f2y_ad_h-f2y_sub_h)/(2*h)
				return np.array([[f1_derivative_x,f1_derivative_y],[f2_derivative_x,f2_derivative_y]])
