import numpy as np
import scipy as scp
from scipy import sparse as sp
from scipy.sparse import linalg as la
import matplotlib
import matplotlib.pyplot as plt
from scipy import ndimage
pi = np.pi 

#Import image here (not yet implemented), define index of refraction
N,M = 300,300
n = np.ones([N+1,M+1]) 

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
s = 0.01 ;

# Define source term 
def normalpdf(x,y):
	return 1.0/np.sqrt(2*pi*pow(s,2))*np.exp(-1.0/(2*pow(s,2))*(pow(x-x0,2)+pow(y-y0,2))) 

vnormalpdf = np.vectorize(normalpdf)
fvect = vnormalpdf(X,Y) ; fvect = fvect.T ;
fvec = fvect.reshape([1,(N+1)*(M+1)])[0] ;

#plt.matshow(fvect) 
# plt.show()

#Construct del^2+k^2*n^2 Matrix
Ax = np.diag(-2.0/pow(dx,2)*np.ones(N+1))+np.diag(1.0/pow(dx,2)*np.ones(N),k=1)+np.diag(1.0/pow(dx,2)*np.ones(N),k=-1)
Ax = sp.csr_matrix(Ax) ;

Ay = np.diag(-2.0/pow(dy,2)*np.ones(M+1))+np.diag(1.0/pow(dy,2)*np.ones(M),k=1)+np.diag(1.0/pow(dy,2)*np.ones(M),k=-1) ; 
Ay = sp.csr_matrix(Ay) ;

A = sp.kron(sp.identity(M+1),Ax)+sp.kron(Ay,sp.identity(N+1)) ;
A = A+sp.diags(pow(k,2)*pow(n.reshape(1,(N+1)*(M+1))[0],2))

#Solve the system... (may add some iterative method here in the future)
E = la.spsolve(A,fvec).reshape(N+1,M+1) 
plt.pcolor(x,y,E,vmin=-0.005,vmax=0.005)
plt.colorbar()
plt.show()
