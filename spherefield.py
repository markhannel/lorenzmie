from sphere_coefficients import sphere_coefficients
from sphericalfield import sphericalfield
import numpy as nmp

def spherefield(x, y, z, a, np,nm = complex(1.3326, 1.5e-8), 
                lamb = 0.447, mpp = 0.135, precision = False,
                cartesian = False):
    """
    Calculate the complex electric field scattered by a sphere illuminated
    by a plane wave linearly polarized in the x direction.
   
    Args:
        x:  [npts] array of pixel coordinates [pixels]
        y:  [npts] array of pixel coordinates [pixels]
        z:  If field is required in a single plane, 
        z is the plane's distance from the sphere's center
        [pixels].
        Otherwise, z is an [npts] array of coordinates.
        a:  radius of sphere [micrometers]
        np: (complex) refractive index of sphere

    Keywords:
        nm:         refractive index of medium (default water)
        lamb:       vacuum wavelength of light (default 0.6328)
        mpp:        micrometers per pixel      (default 0.101)
        precision:  accuracy of scattering coefficients
        k:          scaled wavenumber in medium
        cartesian:  project to cartesian coordinates

    Returns:
        field: [3,npts]
    """
    ab = sphere_coefficients(a, np, nm, lamb)

    if type(precision) != str : 
        # retain first coefficient for bookkeeping
        fac = abs(ab[:, 1])
        w = nmp.where(fac > precision*max(fac))
        w = nmp.concatenate((nmp.array([0]),w[0])) 
        ab =  ab[w,:]
 
    lamb_m = lamb/nmp.real(nm)/mpp # medium wavelength [pixel]

    field = sphericalfield(x, y, z, ab, lamb_m, cartesian=cartesian)

    return field

