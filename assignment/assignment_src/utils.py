import f90nml # requires package: `pip install f90nml`
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

RTYPE=np.float64
ITYPE=np.int32

EARTH_RADIUS=RTYPE(6371000.) # in Units of [m]
DAY_LENGTH  =RTYPE(86400.)     # in units of [s] 
EPSILON     =RTYPE(1e-12) 
PI          =RTYPE(np.pi)
PI2         =2*PI 


def get_long_time() :
    return '[{}] '.format(datetime.utcnow().strftime("%Y%m%d %H:%M:%S.%f")[:-3])
def get_short_time() :
    return '[{}] '.format(datetime.utcnow().strftime("%H:%M:%S.%f")[:-3])

def laplacian2D(g,x,wrk):

    wrk[0,:] =     (x[1,:]  +x[g.nx-1,:] -2*x[0,:])*g.dx2norm 
    for i in range(1,g.nx-1):
        wrk[i,:] = (x[i+1,:]+x[i-1,:]    -2*x[i,:] )*g.dx2norm
    wrk[g.nx-1,:] =(x[0,:]  +x[g.nx-2,:] -2*x[g.nx-1,:])*g.dx2norm 
    
    wrk[:,0]     += (x[:,1]   +x[:,g.ny-1] -2*x[:,0] )    *g.dy2norm
    for j in range(1,g.ny-1):
        wrk[:,j] += (x[:,j+1] +x[:,j-1]    -2*x[:,j] )    *g.dy2norm
    wrk[:,g.ny-1]+= (x[:,0]   +x[:,g.ny-2] -2*x[:,g.ny-1])*g.dy2norm 
        
    return wrk

def der_x(g,x,out):
    out[0,:] =      (x[1,:]   -x[g.nx-1,:])*g.dxnorm
    for i in range(1,g.nx-1):
        out[i,:] =  (x[i+1,:] -x[i-1,:])   *g.dxnorm 
    out[g.nx-1,:] = (x[0,:]   -x[g.nx-2,:])*g.dxnorm 
    return out

def der_y(g,x,out):
    out[:,0] =      (x[:,1]   -x[:,g.ny-1])*g.dynorm
    for j in range(1,g.ny-1):
        out[:,j] =  (x[:,j+1] -x[:,j-1])   *g.dynorm
    out[:,g.ny-1] = (x[:,0]   -x[:,g.ny-2])*g.dynorm 
    return out

def poisson(g,i,o,c_wrk):
    c_wrk=np.fft.rfft2(i)
    i=0
    for j in range(1,int(g.ny/2)+1) :   # we do not touch the mean (j=0 is skipped)
        c_wrk[i,j] = - c_wrk[i,j] / g.lambda_y2[j]

    for i in range(1,g.nx):
        for j in range(int(g.ny/2)+1):
            c_wrk[i,j] = -c_wrk[i,j] / (g.lambda_x2[i] + g.lambda_y2[j])
            
    o=np.real(np.fft.irfft2(c_wrk)) 
    return o


