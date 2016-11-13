import numpy as nmp
from spherefield import spherefield

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
    '''
    nx, ny = map(float, dim)
    x = nmp.tile(nmp.arange(nx), ny)
    y = nmp.repeat(nmp.arange(ny), nx)
    x -= nx/2. + float(rp[0])
    y -= ny/2. + float(rp[1])

    zp = float(rp[2])
    '''

    nx = float(dim[0])
    ny = float(dim[1])
    npts = nx * ny
    x = nmp.arange(npts) % nx
    y = nmp.floor(nmp.arange(npts) / nx)
    x -= nx/2. + float(rp[0])
    y -= ny/2. + float(rp[1])

    zp = float(rp[2])

    if lut == True:
        rho = nmp.sqrt(x**2 + y**2)
        x = nmp.arange(nmp.fix(rho).max()+1)
        y = 0. * x
    
    print x,y

    field = spherefield(x, y, zp, ap, np, nm = nm, 
                        cartesian = True, mpp = mpp, 
                        lamb = lamb, precision = precision)
    print field[0:10,0]
    if alpha: 
        field *= alpha

    k = 2.0*nmp.pi/(lamb/nmp.real(nm)/mpp)
    
    # Compute the sum of the incident and scattered fields, then square.
    field *= nmp.complex(0.,-k*zp)
    field[0,:] += 1.0

    image = nmp.sum(nmp.real(field*nmp.conjugate(field)), axis = 1)

    if lut == True: 
        image = nmp.interpolate(image, rho, cubic=-0.5)

    return image.reshape(ny,nx)
