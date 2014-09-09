#+
# NAME:
#    fitspheredhm
#
# PURPOSE:
#    Measure the radius, refractive index, and three-dimensional
#    position of a colloidal sphere immersed in a dielectric 
#    medium by fitting its digital holographic microscopy (DHM)
#    image to Mie scattering theory.
#
# CATEGORY:
#    Holographic microscopy
#
# CALLING SEQUENCE:
#    params = fitspheredhm(a, p)
#
# INPUTS:
#    a : two-dimensional real-valued DHM image of sphere.
#
#    p : initial guess for fitting parameters.
#      p[0] : xp : x-coordinate of sphere's center [pixels]
#      p[1] : yp : y-coordinate [pixels]
#      p[2] : zp : z-coordinate [pixels]
#      p[3] : ap : sphere's radius [micrometers]
#      p[4] : np : sphere's refractive index
#      p[5] : amplitude : arbitrary.  1 is a reasonable value
#      p[6] : nm : refractive index of medium
#
#      Optional:
#      p[7] : kp : sphere's extinction coefficient
#      p[8] : km : medium's extinction coefficient
#    NOTE: The extinction coefficients are assumed to be positive
#    which is appropriate for absorbing materials in the convention
#    used by SPHERE_COEFFICIENTS.
#
# KEYWORD PARAMETERS:
#    lamb: vacuum wavelength of illumination [micrometers].
#      Default: 0.632816
#
#    mpp: Length-scale calibration factor [micrometers/pixel].
#      Default: 0.135
#
#    precision: Convergence tolerance of nonlinear least-squares fit.
#      Default: 5d-5.
#
#    aplimits: [minap, maxap] limits on ap [micrometers]
#      Default: [0.05, 10.]
#
#    nplimits: [minnp, maxnp] limits on np
#      Default: [1.01 nm, 3.]
#
# KEYWORD FLAGS:
#    deinterlace: Only fit to odd (DEINTERLACE = 1) 
#      or even (DEINTERLACE in range(2) scan lines.  This is useful for analyzing
#      holograms acquired with interlaced cameras.
#
#    fixap: If set, do not allow ap to vary.
#    fixnp: If set, do not allow np or kp to vary.
#    fixnm: If set, do not allow nm or km to vary.
#    fixzp: If set, do not allow zp to vary.
#    fixalpha: If set, do not allow alpha to vary.
#
#    gpu: If set, use GPU acceleration to calculate fields on
#         systems with GPULib installed.
#         Requires NVIDIA GPU with CUDA support.
#
#    object: If set, use a DGGdhmSphereDHM object to compute holograms.
#         This requires GPULib.
#
#    quiet: If set, do not show results of intermediate calculations.
#
# OUTPUTS:
#    params: Least-squares fits for the values estimated in P.
#      params[:, 0]: Fit values.
#      params[:, 1]: Error estimates.
#      NOTE: errors are set to 0 for parameters held constant
#            with the FIX* keyword flags.
#
# RESTRICTIONS:
#    Becomes slower and more sensitive to accuracy of initial
#    guesses as spheres become larger.
#
# PROCEDURE:
#    Uses MPFIT by Craig Marquardt (http://purl.com/net/mpfit/)
#    to minimize the difference between the measured DHM image and 
#    the image computed by SPHEREDHM.
#
# REFERENCES:
# 1. S. Lee, Y. Roichman, G. Yi, S. Kim, S. Yang, A. van Blaaderen, 
#    P. van Oostrum and D. G. Grier, 
#    Chararacterizing and tracking single colloidal particles with 
#    video holographic microscopy, 
#    Optics Express 15, 18275-18282 (2007)
#
# 2. C. B. Markwardt, 
#    Non-linear least squares fitting in IDL with MPFIT, 
#    in Astronomical Data Analysis and Systems XVIII, 
#    D. Bohlender, P. Dowler and D. Durand, eds.
#    (Astronomical Society of the Pacific, San Francisco, 2008).
#
# MODIFICATION HISTORY:
# Written by David G. Grier, New York University, 4/2007.
# 05/22/2007: DGG. Added LAMBDA keyword.
# 05/26/2007: DGG. Revised to use Bohren and Huffman version of
#   SPHEREFIELD.
# 06/10/2007: DGG. Updated for more accurate BH code.
# 09/11/2007: DGG. Made nm a fitting parameter and removed NM keyword.  
#   Replaced FIXINDEX keword with FIXNM.
#   Added FIXNP keyword.  
# 11/03/2007: DGG. Changed FIXRADIUS to FIXAP.  Added FIXZP and FIXALPHA.
# 02/08/2008:  DGG. Treat coordinates as one-dimensional arrays internally
#   to eliminate repeated calls to REFORM.
#   Adopt updated syntax for SPHEREFIELD: separate x, y and z coordinates.
#   Y coordinates were incorrectly cast to float rather than double.
# 02/10/2008: DGG. Added DEINTERLACE. Small documentation fixes.
# 04/16/2008: DGG. Added MPP keyword.  Small documentation fixes.
# 10/13/2008: DGG. Added PRECISION and GPU keywords to make use of new
#   capabilities in SPHEREFIELD.
# 10/17/2008: DGG. Added LUT keyword to accelerate CPU-based fits.  
#   This required setting .STEP = 0.0001 pixel restrictions on
#   the x and y centroids in PARINFO.
# 01/15/2009: DGG. Documentation clean-ups.
# 02/14/2009: DGG. Added APLIMITS and NPLIMITS keywords.
# 03/17/2009: DGG. Added support for complex refractive indexes by
#    accounting for the particle and medium extinction coefficients, 
#    kp and km.
# 03/26/2009: Fook Chiong Cheong, NYU: np and nm should be cast as
#    dcomplex rather than complex when kp or km are non-zero.
# 06/18/2010: DGG. Added COMPILE_OPT.
# 10/20/2010: DGG: Cleaned up alpha code in spheredhm_f.
# 11/30/2010: DGG & FCC: Vary kp and km independently.  Documentation
#    and formatting.
# 11/08/2011: DGG. Removed LUT option: could not guarantee precision.
#    Added OBJECT keyword to compute fits with DGGdhmSphereDHM object
#    for improved efficiency.  Documentation upgrades.
# 11/09/2011: DGG. PRECISION keyword now corresponds to FTOL in MPFIT.
# 04/17/2012: DGG. Fixed deinterlace code for object-based fits for
#    centers aligned with grid but outside of field of view.
# 05/03/2012: DGG. Updated parameter checking.
#
# Copyright (c) 2007-2012, David G. Grier and Fook Chiong Cheong
#
# UPDATES:
#    The most recent version of this program may be obtained from
#    http://physics.nyu.edu/grierlab/software.html
# 
# LICENSE:
#    This program is free software# you can redistribute it and/or
#    modify it under the terms of the GNU General Public License as
#    published by the Free Software Foundation# either version 2 of the
#    License, or (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful, 
#    but WITHOUT ANY WARRANTY# without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program# if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
#    02111-1307 USA
#
#    If the Internet and WWW are still functional when you are using
#    this, you should be able to access the GPL here: 
#    http://www.gnu.org/copyleft/gpl.html
#-

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

   field = spherefield(xx, yy, zp, ap, np, \
                    nm = nm, \
                    lamb = lamb, \
                    mpp = mpp, \
                    cartesian = 'cartesian', \
                    gpu ='')


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
                 lamb = 0.632816,       # wavelength of light [micrometers]
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
                 gpu = 0,               # use GPU acceleration
                 object = 0,            # use DGGdhmSphereDHM object (TURNED OFF)
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

    if object == 'object':
       #np = complex(p0[4], (nparams >= 8) ? p0[7] : 0)
       #nm = complex(p0[6], (nparams >= 8) ? p0[8] : 0)
       obj = DGGdhmSphereDHM(dim = [nx, ny], #$
                             lamb = 0.326, #$
                             mpp = 0.135, #$
                             rp = p0[0:2], #$
                             ap = p0[3], #$
                             nm = nm, #$
                             np = np, #$
                             alpha = p0[5], #$
                             deinterlace = 0 #$
                             )

   #if ~isa(obj, 'DGGdhmSphereDHM') :
	#message, 'could not create a DGGdhmSphereDHM object', /inf
	#return -1

       aa = float(a)
       #if len(deinterlace) > 0:
           #w = where((lindgen(ny) mod 2) == (deinterlace mod 2), ny)
           #aa = aa[w, :]

       err = array([5. for i in xrange(npts)]) # FIXME this works but could be made rigorous
      
       p = mp.mpfit(spheredhm_objf, obj, aa, err, p0, 
                       parinfo = parinfo, ftol = precision)
       
    else:
       x = tile(arange(nx,dtype = float),npts/nx)  # coordinates of pixels
       y = repeat(arange(ny,dtype = float),npts/ny)
       aa = array(a,float).flatten()
                
       #if len(deinterlace) > 0:
        #w = where((y mod 2) == (deinterlace mod 2), npts)
        #x = x[w]
        #y = y[w]
        #aa = aa[w]

       err = array([5.]*npts)    # FIXME this works but could be made rigorous

       x -= double(nx) / 2.0
       y -= double(ny) / 2.0

       # parameters passed to the fitting function
       argv = {'x':x, 'y':y, 'z':aa,'lamb':lamb, 'mpp':mpp, 'err':err}

       # perform fit
       p = mp.mpfit(spheredhm_f, p0, functkw = argv, parinfo = parinfo, ftol = precision)

    #if status <= 0 : 
    #  message, errmsg, /inf
    #  return -1
    
    # failure?
    #if len(p) == 1 :
    #  message, "MPFIT2DFUN did not return a result",  /inf
    #  return -1

    # success
    # rescale fit uncertainties into error estimates
    #dp = perror * sqrt(bestnorm/dof)

    return transpose(p.params)
