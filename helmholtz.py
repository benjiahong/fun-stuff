import numpy as np
import scipy as scip
from numpy import random 
import matplotlib
import matplotlib.pyplot as plt
from scipy import ndimage
pi = np.pi 
N,M = 200,100
n = np.ones(shape = (N,M)) 

for i in range(0,N):
	for j in range (0,M):
		if n[i,j] == 0:
			n[i,j] = float(np.random.randn(1))

L,H = 1.0,1.0 ;
k = (2*pi*2.4e9)/3e8
dx = L/N ; dy = H/M ;
x = np.linspace(0.0,L,num = N+1) ; y = np.linspace(0.0,H,num = M+1) ;
[X,Y] = np.meshgrid(x,y) ;
x0 = L/2; y0 = H/2

def normalpdf(x,y):
	return 1000.0/np.sqrt(2*pi*pow(0.05,2))*np.exp(-1.0/(2*pow(0.05,2))*(pow(x-x0,2)+pow(y-y0,2))) 

vnormalpdf = np.vectorize(normalpdf)
fvect = vnormalpdf(X,Y) ; fvect = fvect.T ;
fvec = fvect.reshape([(N+1)*(M+1),1]) 

plt.matshow(fvect) 
plt.show()

