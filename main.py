import numpy as np 
from qg import QG_MODEL,INTEGRATOR,GRID


def main():
    
    m = QG_MODEL()                       # initialize the model class
    m.initialize()                       # initialize the simulation 

    integrator = INTEGRATOR(m)           # initialize time integration
    integrator.time_integrate(m.g,m.mode)# run the simulation 

    if m.mode == 'exponential' :
        # integrating x yields exponential function
        # use this from t=0 to t=t1 to check accuracy of time integration
        m.log_line("Numerical error at final state: {}".format(m.data.zeta[0,0]-m.zeta0*np.exp(m.rtime)))
        
    m.finalize(m.mode)                   # clean up 
    return 0

if __name__ == "__main__":
    
    # test poisson solver
    # def poisson(g,i,o,c_wrk):
    # m=CFQG()
    # m.initialize()
    # kx=0
    # ky=4
    # for j in range(m.g.ny):
    #    jarg = PI/2.*m.g.y[j]/m.g.ly
    #    m.data.zeta[:,j]=m.zeta0*np.cos(kx*PI2*m.g.x/m.g.lx)*np.cos(ky*jarg) 
    # m.data.wrk[:,:,0] = poisson(m.g,m.data.zeta,m.data.wrk[:,:,0],m.data.c_wrk)
    # print(m.zeta0/k2/np.amax(m.data.wrk[:,:,0]))
    
    main()  
