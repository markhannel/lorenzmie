import numpy as nmp
from spherefield import *

def spheredhm(rp, ap, np, nm, dim, mpp = 0.135, lamb = .447, alpha = False, 
              precision = False,  lut = False):
    """
    Compute holographic microscopy image of a sphere immersed in a transparent 
    medium.

    Args:
        rp  : [x, y, z] 3 dimensional position of sphere relative to center
              of image.
        ap  : radius of sphere [micrometers]
        np  : (complex) refractive index of sphere
        nm  : (complex) refractive index of medium
        dim : [nx, ny] dimensions of image [pixels]

    NOTE: The imaginary parts of the complex refractive indexes
    should be positive for absorbing materials.  This follows the
    convention used in SPHERE_COEFFICIENTS.

    Keywords:
        precision: 
        alpha: fraction of incident light scattered by particle.
            Default: 1.
        lamb:  vacuum wavelength of light [micrometers]
        mpp: micrometers per pixel
        precision: relative precision with which fields are calculated.
    
    Returns:
        dhm: [nx, ny] holographic image                
    """

    nx, ny = map(float, dim)
    x = nmp.tile(nmp.arange(nx), ny)
    y = nmp.repeat(nmp.arange(ny), nx)
    x -= nx/2. + float(rp[0])
    y -= ny/2. + float(rp[1])

    zp = float(rp[2])

    if lut == True:
        rho = sqrt(x**2 + y**2)
        x = arange(fix(rho).max()+1)
        y = 0. * x
        
    field = spherefield(x, y, zp, ap, np, nm = nm, 
                        cartesian = True, mpp = mpp, 
                        lamb = lamb, precision = precision, 
                        gpu = gpu)

    if len(alpha) == 1 : 
        field *= alpha

    # scattered intensity
    dhm = sum(real(field * conj(field)), 1)

    # interference between scattered wave and incident plane wave
    k = 2.0*pi/(lamb/real(nm)/mpp)
    dhm += 2.0 * real(field[:, 0] * exp(complex(0, -k*zp)))

    # intensity of plane wave
    dhm += 1.0

    if lut == True: 
        dhm = interpolate(dhm, rho, cubic=-0.5)

    return dhm.reshape(ny,nx)
