from numpy import *
import mpfit as mp
from spherefield import spherefield

class spheredhm_objf:
   def __init__(self, p, obj):
      # p[0] : xp         x position of sphere center
      # p[1] : yp         y position of sphere center
      # p[2] : zp         z position of sphere center
      # p[3] : ap         radius of sphere
      # p[4] : np         real part of sphere's refractive index
      # p[5] : alpha  
      # p[6] : nm         real part of medium's refractive index
      #
      # Optional:
      # p[7] : kp         imaginary part of sphere's refractive index
      # p[8] : km         imaginary part of medium's refractive index


      # THE BELOW PARAMETERS ARE INCORRECT.  NEED AN IMPLEMENTATION OF
      # DGGDHMSPHEREDHM OBJECT!!!!


      self.nparams = len(p)
      self.rp      = p[0:2] 
      self.p       = p[3] 
      self.alpha   = p[5]
      self.np      = p[4]
      self.nm      = p[6]
      if self.nparams > 8: self.np += nmp.complex(0,p[7])
      if self.nparams > 9: self.nm += nm.complex(0,p[8])
#   return obj.hologram


def spheredhm_f(p, x,y,z,lamb,mpp,err,fjac = None,**kwargs):
   # p[0] : xp         x position of sphere center
   # p[1] : yp         y position of sphere center
   # p[2] : zp         z position of sphere center
   # p[3] : ap         radius of sphere
   # p[4] : np         real part of sphere's refractive index
   # p[5] : alpha  
   # p[6] : nm         real part of medium's refractive index
   #
   # Optional:
   # p[7] : kp         imaginary part of sphere's refractive index
   # p[8] : km         imaginary part of medium's refractive index
   
   xx = x - p[0]
   yy = y - p[1]
   zp = p[2]
   ap = p[3]
   np = p[4]
   alpha = p[5]
   nm = p[6]
   if len(p) >= 8 : #$
      np = complex(np, p[7])      # sphere's complex refractive index
   if len(p) >= 9 : #$ 
      nm = complex(nm, p[8])      # medium's complex refractive index


   field = spherefield(xx, yy, zp, ap, np, nm = nm, lamb = lamb, mpp = mpp, 
                       cartesian = True)

   if alpha:
      field *= alpha

   # interference between light scattered by the particle
   # and a plane wave polarized along x and propagating along z
   lamb_m = lamb / real(nm) / mpp
   k = 2.0 * pi / lamb_m  

   dhm = 1.0 + \
       2.0 *(field[:, 0] * exp(complex(0, -k*zp))).real + \
       sum((field * conj(field)), axis=1)
   
   #w =  where(finite(dhm, /nan), nbad)
   #if nbad != 0 : stop
   status = 0
   return [status, abs(dhm-z)/err]

def fitspheredhm(a,                     # image
                 p0,                    # starting estimates for parameters
                 aplimits = [0.05,10],  # limits on ap [micrometers]
                 nplimits = [],   # limits on np
                 lamb = 0.447   ,       # wavelength of light [micrometers]
                                        # (HeNe wavelength)
                 mpp = 0.135,           # micrometers per pixel
                 fixnp = 0,             # fix particle refractive index
                 fixnm = 0,             # fix medium refractive index
                 fixap = 0,             # fix particle radius
                 fixzp = 0,             # fix particle axial position
                 fixalpha = 0,          # fix illumination
                 deinterlace = '',  
                 precision = 5E-5,      # precision of convergence
                                        #(Default precision keeps all scattering
                                        # Coefficients)
                 quiet = 'quiet'):      # don't print diagnostics

    sz = a.shape
    nx = sz[0]
    ny = sz[1]
    npts = nx*ny
    nparams = len(p0)
    
    parinfo = [{'fixed':0, 'limited':[0,0], 'limits':[0.,0.], 'step':0}\
                  for i in xrange(nparams)]
 

    ## Restrictions on fitting parameters
    # xp and yp: overly small steps sometimes prevent convergence
    parinfo[0]['step'] = 0.0001
    parinfo[1]['step'] = 0.0001
    # zp: No restrictions
    # 
    # ap: Radius must be positive
    parinfo[3]['limited'][0] = 1
    parinfo[3]['limited'][1] = 1
    if len(aplimits) == 2 : 
       parinfo[3]['limits'] = aplimits
    # np: Refractive index of particle
    parinfo[4]['limited'][0] = 1
    parinfo[4]['limits'][0] = 1.01*p0[6] # FIXME what about low-index particles?
    parinfo[4]['limited'][1] = 1
    parinfo[4]['limits'][1] = 3.0      # a bit more than titania
    if len(nplimits) == 2 : 
       parinfo[4]['limits'] = nplimits
    # alpha: Illumination at particle
    parinfo[5]['limited'][0] = 1
    parinfo[5]['limits'][0] = 0.       # cannot be negative
    # nm: Refractive index of medium: No restrictions
    # Flags to prevent parameters from being adjusted
    parinfo[2]['fixed'] = fixzp
    parinfo[3]['fixed'] = fixap
    parinfo[4]['fixed'] = fixnp
    parinfo[5]['fixed'] = fixalpha
    parinfo[6]['fixed'] = fixnm
    if nparams >= 8 :
       parinfo[7]['fixed'] = fixnp
       parinfo[7]['limited'][:] = 1
       parinfo[7]['limits'] = [0.0, 5.0]
    if nparams >= 9 :
       parinfo[8]['fixed'] = fixnm
       parinfo[8]['limited'][:] = 1
       parinfo[8]['limits'] = [0.0, 5.0]

    # errors from fit
    perror = zeros(nparams)


    x = tile(arange(nx,dtype = float),npts/nx)  # coordinates of pixels
    y = repeat(arange(ny,dtype = float),npts/ny)
    aa = array(a,float).flatten()
                

    err = array([5.]*npts)    # FIXME this works but could be made rigorous

    x -= double(nx) / 2.0
    y -= double(ny) / 2.0

    # parameters passed to the fitting function
    argv = {'x':x, 'y':y, 'z':aa,'lamb':lamb, 'mpp':mpp, 'err':err}

    # perform fit
    p = mp.mpfit(spheredhm_f, p0, functkw = argv, parinfo = parinfo, ftol = precision)

    return transpose(p.params)
