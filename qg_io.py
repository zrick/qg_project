from netCDF4 import Dataset


class QG_IO:
    def __init__(self,fname,grid):
        self.fname=fname
        self.g=grid
        self.niter=0 

    def initialize(self,mode):
        print('INITIALIZING DATASET',self.fname) 
        self.f_h = Dataset(self.fname,'w')
        f=self.f_h
        g=self.g
        f.createDimension('x',g.nx)
        f.createDimension('y',g.ny)
        f.createDimension('t',None)

        vx=f.createVariable('x','f','x')
        vx.long_name='zonal distance'
        vx.standard_name='x'
        vx.units='m'
        f.variables['x'][:] = g.x[:]
        
        vy=f.createVariable('y','f','y')
        vy.long_name='meridional distance'
        vy.standard_name='y'
        vy.units='m'
        f.variables['y'][:] = g.y[:]

        vt=f.createVariable('t','f','t')
        vt.long_name='time'
        vt.standard_name='t'
        vt.units='s'

        vit=f.createVariable('it','f','t')
        vit.long_name='iteration'
        vit.standard_name='it'
        vit.units='1'
        
        if mode == 'rossby' or mode == 'exponential' :
            vzeta=f.createVariable('zeta','f',('t','x','y',))
        elif mode == 'two_layer':

            f.createDimension('z',2)
            vz = f.createVariable('z','f',('z',))
            vz.long_name='height [in p-system]'
            vz.standard_name='z'
            vz.units='hPa'
            f.variables['z'][:] = [250,750] 

            vzeta=f.createVariable('zeta','f',('t','z','x','y',))
        vzeta.long_name='Vorticity'
        vzeta.standard_name='zeta'
        vzeta.units='s^{-1}'
        

        return 
            
    def dump_it(self,it,t,data):
        f=self.f_h
        i=self.niter
        f.variables['t'][i] = t
        f.variables['it'][i] = it
        f.variables['zeta'][i] = data 
        
        self.niter += 1 
        
    def finalize(self):
        self.f_h.close()
        
        
        
        
        
    
