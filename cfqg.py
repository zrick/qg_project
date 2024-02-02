import f90nml # requires package: `pip install f90nml`
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from utils import get_long_time, get_short_time, poisson,der_x,der_y
from qg_io import QG_IO 
RTYPE=np.float64
ITYPE=np.int32

EARTH_RADIUS=RTYPE(6371000.) # in Units of [m]
DAY_LENGTH  =RTYPE(86400.)     # in units of [s] 
EPSILON     =RTYPE(1e-12) 
PI          =RTYPE(np.pi)
PI2         =2*PI 

class GRID:
    def __init__(self,nx,ny,lx=RTYPE(1.),ly=RTYPE(1.)):
        self.nx=nx
        self.ny=ny
        self.nxm=nx-1
        self.nym=ny-1
        self.nxp=nx+1
        self.nyp=ny+1
        self.lx=RTYPE(lx)
        self.ly=RTYPE(ly) 
        self.dx=lx/RTYPE(nx)
        self.dy=ly/RTYPE(ny)
        self.x=np.arange(0,lx,self.dx,dtype=RTYPE)
        self.y=np.arange(0,ly,self.dy,dtype=RTYPE)
        self.dx2norm = 1./(self.dx**2)
        self.dy2norm = 1./(self.dy**2)
        self.dxnorm = 1./(2*self.dx)
        self.dynorm = 1./(2*self.dy)

        self.lambda_x  = np.zeros(nx,dtype=RTYPE)
        self.lambda_y  = np.zeros(int(ny/2)+1,dtype=RTYPE)
        self.lambda_x2 = np.zeros(nx,dtype=RTYPE)
        self.lambda_y2 = np.zeros(int(ny/2)+1,dtype=RTYPE)

        
        for i in range(int(nx/2)+1):
            ilx_r = PI2*RTYPE(i)/lx
            self.lambda_x[i]  = ilx_r
            self.lambda_x2[i] = ilx_r ** 2
        for i in range(int(nx/2)+1,nx):
            nxmilx_r=PI2*RTYPE(nx-i)/lx
            self.lambda_x[i]  = PI2*nxmilx_r 
            self.lambda_x2[i] = nxmilx_r ** 2 
        for j in range(int(ny/2)+1):
            jlx_r = PI2*RTYPE(j)/ly 
            self.lambda_y[j]  = PI2*jlx_r
            self.lambda_y2[j] = jlx_r ** 2 

class DATA:
    def __init__(self,grid,mode):
        if mode.lower() == 'rossby': 
            self.zeta=np.zeros([grid.nx,grid.ny],dtype=RTYPE) 
            self.dzeta=np.zeros([grid.nx,grid.ny],dtype=RTYPE)
            self.wrk=np.zeros([grid.nx,grid.ny,4],dtype=RTYPE)
            self.c_wrk=np.zeros([grid.nx,int(grid.ny/2)+1],dtype=np.complex64) 
        if mode.lower() == 'exponential':
            self.zeta=np.zeros([grid.nx,grid.ny],dtype=RTYPE)
            self.dzeta=np.zeros([grid.nx,grid.ny],dtype=RTYPE) 
        return 
        
class INTEGRATOR:
    def __init__(self,model):
        # fourth-order five-stage low-storage Runge--Kutta method
        # (cf. Williamson 1980, Journal Comp. Phys. vol. 35)
        self.model=model
        self.nstep = 5
        self.kdt = np.zeros([self.nstep],dtype=RTYPE); kdt=self.kdt
        self.ktime=np.zeros([self.nstep],dtype=RTYPE); ktime=self.ktime
        self.kco  =np.zeros([self.nstep],dtype=RTYPE); kco=self.kco 
        kdt[0]=RTYPE(1432997174477.0)/RTYPE(9575080441755.0)
        kdt[1]=RTYPE(5161836677717.0)/RTYPE(13612068292357.0)
        kdt[2]=RTYPE(1720146321549.0)/RTYPE(2090206949498.0)
        kdt[3]=RTYPE(3134564353537.0)/RTYPE(4481467310338.0)
        kdt[4]=RTYPE(2277821191437.0)/RTYPE(14882151754819.0)

        ktime[0]=RTYPE(0.0)
        ktime[1]=RTYPE(1432997174477.0)/RTYPE(9575080441755.0)
        ktime[2]=RTYPE(2526269341429.0)/RTYPE(6820363962896.0)
        ktime[3]=RTYPE(2006345519317.0)/RTYPE(3224310063776.0)
        ktime[4]=RTYPE(2802321613138.0)/RTYPE(2924317926251.0)

        kco[0]=RTYPE(-567301805773.0)/RTYPE(1357537059087.0)
        kco[1]=RTYPE(-2404267990393.0)/RTYPE(2016746695238.0)
        kco[2]=RTYPE(-3550918686646.0)/RTYPE(2091501179385.0)
        kco[3]=RTYPE(-1275806237668.0)/RTYPE(842570457699.0)
        kco[4]=RTYPE(0.0)
        
        return

    def time_step(self,grid,rhs):
        m=self.model
        for istep in range(self.nstep) :
            dte=self.dtime*self.kdt[istep]
            m.data.dzeta=m.data.dzeta + rhs(grid,m.data.zeta,m.data.dzeta)
            m.data.zeta = m.data.zeta+ dte*m.data.dzeta
            m.data.dzeta=self.kco[istep]*m.data.dzeta
        return

    def time_courant(self):
        m=self.model
        if m.step_time > 0:
            m.dtime=m.step_time 
        else:
            g=m.g
            CFL=-m.step_time
            d_eff = np.amin([g.dx/self.umax,g.dy/self.vmax])
            m.dtime=CFL*d_eff

        if m.dtime < 0:
            m.log_line('ERROR: dtime<0; not integrating backwards')
            print('ERROR - check {}'.format(m.log_file))
            quit() 
            
        return m.dtime 

    
    def time_integrate(self,grid,mode):
        m=self.model
        end=m.end_time

        if mode == 'exponential':
            rhs = self.exponential
        elif mode == 'rossby':
            rhs = self.rossby
            # make an init call to calculate u and psi for estimation of the initial time step
            rhs(m.g,m.data.zeta,m.data.dzeta)
        else:
            quit()

        #initialize the time step 
        self.dtime = self.time_courant()
            
        m.log_iteration(header=True)
        m.log_line('Starting time integration at time {} '.format(m.rtime))

        plt.imshow(self.model.data.zeta.transpose())
        plt.colorbar()
        plt.show()
        # plt.close('all') 
        
        while( m.rtime < m.end_time - EPSILON ):
            
            if m.rtime + self.dtime > m.end_time - EPSILON:
                self.dtime = m.end_time - m.rtime
                m.dtime = self.dtime 

            self.time_step(grid,rhs)

            m.itime += 1 
            m.rtime += self.dtime

            if m.itime % m.iteralog == 0:
                m.log_iteration()
                plt.imshow(self.model.data.zeta.transpose())
                plt.colorbar()
                plt.show()
                plt.close('all') 

            # determine time-lapse for next step (step zero is done above) 
            self.dtime = self.time_courant()

        # log last iteration if not done yet 
        if m.itime % m.iteralog != 0:
            m.log_iteration()
            
        m.log_line('Reached end time {} ({} days)'.format(m.rtime,m.rtime/DAY_LENGTH))
        m.log_line('Final dt={}'.format(self.dtime))
        m.log_line('Finished time integration in {} iterations'.format(m.itime))
        
    def rossby(self,g,zeta,out):
        # tendency terms are *added* to out
        
        psi = self.model.data.wrk[:,:,0]
        u =   self.model.data.wrk[:,:,1]
        v =   self.model.data.wrk[:,:,2]
        wrk1= self.model.data.wrk[:,:,3]
        c_wrk=self.model.data.c_wrk

        # calculate stream-function and velocity 
        psi = poisson(g,zeta,psi,c_wrk)

        # advection terms + beta-effect
        u= der_y(g,psi,u); u=-u   # u=-dpsi/dy
        out[:,:] = out     -(u+self.model.Um)*der_x(g,zeta,wrk1) # -(u'+Um)(dzeta'/dx)
        v= der_x(g,psi,v)         # v= dpsi/dx 
        out[:,:] = out[:,:]-v[:,:]*           der_y(g,zeta,wrk1) # - v' (dzeta'/dy) 
        out = out - self.model.beta*v                            # -beta * v'

        # forcing
        if self.model.forc_type == 'none':
            out = out # nothing to do
        elif self.model.forc_type == 'topography':
            # topographic forcing
            nx = g.nx
            xref = self.model.g.x[int(nx/2)] 
            for i in range(nx): 
                relarg = (20.*(g.x[i] - xref)/g.lx)**2
                wrk1[i,:] = np.exp(-1./(1-relarg)) if relarg < 1. else 0.
            out = out + (self.model.forc_strength*self.model.zeta0)*wrk1
            out = out - 1e-6*zeta 
        else :
            msg='ERROR: Forcing type \'{}\' not implemented'.format(self.model.forc_type)
            print(msg)
            self.model.log_line(msg)
            quit() 
        
        
        # save maximum of u and v for Courant-Friedrichs-Levy (CFL) criterion
        self.umax = np.amax(u+self.model.Um)
        self.vmax = np.amax(v) 
        
        return out  

    def exponential(self,g,x,dummy=None):
        return(x) 


class CFQG:
    status = 0

    def __init__(self,ini_file='cfqg.ini'):
        with open(ini_file) as nml_file:
            self.nml = f90nml.read(nml_file)

    def line_to_file(self,line,f_h,t_format='short') :
        if t_format == 'short': 
            f_h.write("{}{}\n".format(get_short_time(),line) )
        elif t_format == 'long':
            f_h.write("{}{}\n".format(get_long_time(),line) )
        else :
            f_h.write("{}{}\n".format('',line) )
            
    def iter_line(self,line,t_format='short'):
        self.line_to_file(line,open(self.iter_file,'a'),t_format)
    
    def log_line(self,line,t_format='short'):
        self.line_to_file(line,open(self.log_file,'a'),t_format)
        
    def nl_parameter(self,section,name,default=''):
        n=self.nml
        if section in n.keys():
            if name in n[section].keys():
                return n[section][name]
        return default

    def log_iteration(self,header=False):
        d=self.data
        var_min=np.amin(d.zeta)
        var_max=np.amax(d.zeta)
        var_avg=np.average(d.zeta)
        var_std=np.std(d.zeta)
        day_tim=self.rtime/86400. 
        # calculate position of first maximum (indicates phase speed)
        imin=-1
        imax=-1
        if self.g.nx > 1: 
            j = int(RTYPE(self.g.ny)/3.)
            i=0
            while ( d.zeta[i+1,j] > d.zeta[i,j] and i<self.g.nxm):
                i+=1
            if i == self.g.nx-1:
                imax=-1
            else:
                imax=i

            i=0
            while ( d.zeta[i+1,j] < d.zeta[i,j] and i<self.g.nxm):
                i+=1
            if i == self.g.nx-1:
                imin=-1
            else:
                imin=i
                
        if header == True :
            self.iter_line('#==========================================================================')
            self.iter_line('#  ITIME    RTIME    DTIME    VARAVG    VARMIN    VARMAX   VARSTD XLOC NLOC')
            self.iter_line('#==========================================================================')

        self.iter_line('{0:8d} {1:8.3g} {2:8.3g} {3:9.3g} {4:9.3g} {5:9.3g} {6:8.3g} {7:4d} {8:4d}'.format(self.itime,day_tim,self.dtime,var_avg,var_min,var_max,var_std,imin,imax))

    def initialize(self):
        self.mode = self.nl_parameter('Main','mode','rossby').lower()
        md=self.mode
        self.log_file = self.nl_parameter('Main','log_name','{}.log'.format(md)).lower()
        print('LOGGING TO FILE:    {}'.format(self.log_file) )
        self.iter_file = self.nl_parameter('Main','log_name','{}.iter'.format(md)).lower()
        print('ITERATIONS TO FILE: {}'.format(self.iter_file) )
        self.log_line('---')
        self.log_line('STARTING Quasi-Geostrophic Model','long')
        

        nx=ITYPE(self.nl_parameter(md,'Nx','128'))
        ny=ITYPE(self.nl_parameter(md,'Ny','128'))

        self.log_line('mode = {}'.format(md)) 
        self.rtime=RTYPE(0.)
        self.itime=ITYPE(0) 

        self.end_time = RTYPE(self.nl_parameter(md,'end','1'))
        self.step_time= RTYPE(self.nl_parameter(md,'step_time','0.1')) 
        self.iteralog = int(self.nl_parameter(md,'itera_log','1')) 

        self.kx=RTYPE(self.nl_parameter(md,'kglob_x','0'))
        self.ky=RTYPE(self.nl_parameter(md,'kglob_y','0'))
        self.zeta0=RTYPE(self.nl_parameter(md,'zeta0','1e-5'))
        self.end_time=RTYPE(self.nl_parameter(md,'End',10))
        if md == 'rossby': 
            self.phi0=RTYPE(self.nl_parameter(md,'phi0','45'))/180*PI
            self.Um =RTYPE(self.nl_parameter(md,'Um','1'))

            #ROTATION 
            self.lx=np.cos(self.phi0)*PI2*EARTH_RADIUS
            self.ly=self.lx * RTYPE(ny)/RTYPE(nx)
            self.f0 = RTYPE(2)*PI2 / DAY_LENGTH * np.sin(self.phi0)
            self.beta =RTYPE(2)*PI2/ DAY_LENGTH * np.cos(self.phi0) / EARTH_RADIUS       # df/dy = df/dphi * dphi/dy = 1/R (df/dphi)
            self.log_line('CORIOLIS PARAMETER: {}'.format(self.f0))
            self.log_line('BETA PARAMETER:     {}'.format(self.beta)) 

            #FORCING
            self.forc_type = self.nl_parameter(md,'forc_type','none')
            if self.forc_type != 'none':
                self.forc_strength = self.nl_parameter(md,'forc_strength','0.01') 
            
        elif md == 'exponential':
            self.end_time=RTYPE(self.nl_parameter(md,'End',10)) 
            self.lx=1.
            self.ly=1. 
            self.data=DATA(self.g,md)
            self.zeta0=RTYPE(self.nl_parameter(md,'zeta0','1') )
            
        else :
            self.log_line('ERROR - unsupported mode')
            self.finalize() 

        self.g=GRID(nx,ny,self.lx,self.ly)
        self.log_line('Initialized grid {}x{}'.format(self.g.nx,self.g.ny))
        self.log_line('Grid Spacing dx={}; dy={}'.format(self.g.dx,self.g.dy))
        g=self.g

        self.data=DATA(self.g,md)
        self.log_line('Allocated Data for mode {}'.format(md))
        
        self.data.zeta[:,:] = self.zeta0 


        itype = self.nl_parameter(md,'ini_type','harmonic')
        self.log_line('Initialization mode: {}'.format(md))
        if itype.lower() == 'harmonic' :
            for i in range(g.nx):
                if self.kx != 0: 
                    self.data.zeta[i,:]*= np.cos(PI2*self.kx*g.x[i]/self.lx)
                if self.ky != 0:
                    self.data.zeta[i,:]*= np.sin(PI2*self.ky*g.y/self.ly)
        elif itype.lower() == 'singular' :
            [xr,yr] = [g.x[int(nx/2)], g.y[int(ny/2)]];
            for i in range(g.nx):
                relarg = (self.kx/self.lx * (g.x[i]-xr))**2
                v = np.exp(-1./(1-relarg),dtype=RTYPE) if relarg < 1. else 0.
                self.data.zeta[i,:] *= v

            for j in range(g.ny):
                relarg = (self.ky/self.ly * (g.y[j]-yr))**2 
                v = np.exp(-1./(1-relarg),dtype=RTYPE) if relarg < 1. else 0. 
                self.data.zeta[:,j] *= v
        elif itype.lower() == 'random' :
            self.data.zeta[:,:] *= (np.random.rand(nx,ny)-0.5)
        elif itype.lower() == 'none':
            self.data.zeta[:,:] = np.float64(0.)
        self.log_line('Model initialization finished for mode {}'.format(md))
        self.log_line('---')
        
    def finalize(self,mode):
        self.log_line('---')
        self.log_line('FINISHED Quasi-Geostrophic model','long')
        self.log_line('---')
        quit() 
