import numpy as nmp

def check_if_numpy(x, char_x):
    ''' checks if x is a numpy array '''
    if type(x) != nmp.ndarray:
        print char_x + ' must be an numpy array'
        return False
    else:
        return True


def lm_angular_spectrum(geom, ab, lamb, nm, r, z = 0):
    """
    Calculate the angular spectrum of the electric field strength factor given
    scattering coefficients.

    Args:
        sx: [npts] array of pixel coordinates
        sy: [npts] array of pixel coordinates
        f: [float] distance to focal plane [pixels]
        ab: [2,nc] array of a and b scattering coefficients, where
            nc is the number of terms required for convergence.
    Keywords:
        lamb: wavelength of light in medium [pixels]
        cartesian: if True, field is expressed as (x,y,z) else (r, theta, phi)

    Returns:
        field: [3,npts] scattered electric field strength factor
    """
    '''
    # Check that inputs are numpy arrays
    for var, char_var in zip([sx,sy,ab], ['sx', 'sy', 'ab']):
        if check_if_numpy(var, char_var) == False:
            print 'sx, sy and ab must be numpy arrays'
            return None

    if sx.shape != sy.shape:
        print 'sx has shape {} while sy has shape {}'.format(sx.shape, sy.shape)
        print 'and yet their dimensions must match.'
        return None
    '''

    sx = geom.xx.ravel()
    sy = geom.yy.ravel()
    
    npts = len(sx)
    nc = len(ab[:,0])-1     # number of terms required for convergence

    k = 2.0*nmp.pi*nm/ lamb # wavenumber in medium [pixel^-1]

    # Compute relevant coordinates.
    unit_rho_sq = sx**2+sy**2
    costheta = nmp.sqrt(1. - unit_rho_sq)
    costheta /= nmp.sqrt(1 + (2*z/r)*nmp.sqrt(1 - unit_rho_sq))
    sintheta = nmp.sqrt(1 - costheta**2)

    cosphi   = sx/sintheta
    sinphi   = sy/sintheta

    kr = k*(r*nmp.sqrt(1 + 2*(z/r)*nmp.sqrt(1.-unit_rho_sq)))  # reduced radial coordinate

    # starting points for recursive function evaluation ...
    # ... Riccati-Bessel radial functions, page 478
    sinkr = nmp.sin(kr)
    coskr = nmp.cos(kr)

    # FIXME (MDH): What about particles below the focal plane?
    xi_nm2 = coskr + 1.0j*sinkr # \xi_{-1}(kr) 
    xi_nm1 = sinkr - 1.0j*coskr # \xi_0(kr)    

    # ... angular functions (4.47), page 95
    pi_nm1 = 0.0                    # \pi_0(\cos\theta)
    pi_n   = 1.0                    # \pi_1(\cos\theta)
 
    # storage for vector spherical harmonics: [r,theta,phi]
    Mo1n = nmp.zeros([3,npts],complex)
    Ne1n = nmp.zeros([3,npts],complex)

    # storage for scattered field
    Es = nmp.zeros([3,npts],complex)
        
    # Compute field by summing multipole contributions
    for n in xrange(1, nc+1):

        # upward recurrences ...
        # ... Legendre factor (4.47)
        # Method described by Wiscombe (1980)
        swisc = pi_n * costheta 
        twisc = swisc - pi_nm1
        tau_n = pi_nm1 - n * twisc  # -\tau_n(\cos\theta)

        # ... Riccati-Bessel function, page 478
        xi_n = (2.0*n - 1.0) * xi_nm1 / kr - xi_nm2    # \xi_n(kr)

        # vector spherical harmonics (4.50)
        #Mo1n[0,:] = 0               # no radial component
        Mo1n[1,:] = pi_n * xi_n     # ... divided by cosphi/kr
        Mo1n[2,:] = tau_n * xi_n    # ... divided by sinphi/kr

        dn = (n * xi_n)/kr - xi_nm1
        Ne1n[0,:] = n*(n + 1.0) * pi_n * xi_n # ... divided by cosphi sintheta/kr^2
        Ne1n[1,:] = tau_n * dn      # ... divided by cosphi/kr
        Ne1n[2,:] = pi_n  * dn      # ... divided by sinphi/kr

        # prefactor, page 93
        En = 1.0j**n * (2.0*n + 1.0) / n / (n + 1.0)

        # the scattered field in spherical coordinates (4.45)
        Es += En * (1.0j * ab[n,0] * Ne1n - ab[n,1] * Mo1n)
 
        # upward recurrences ...
        # ... angular functions (4.47)
        # Method described by Wiscombe (1980)
        pi_nm1 = pi_n
        pi_n = swisc + (n + 1.0) * twisc / n

        # ... Riccati-Bessel function
        xi_nm2 = xi_nm1
        xi_nm1 = xi_n

    # geometric factors were divided out of the vector
    # spherical harmonics for accuracy and efficiency ...
    # ... put them back at the end.
    Es[0,:] *= cosphi * sintheta / (k**2*r)
    Es[1,:] *= cosphi / k
    Es[2,:] *= sinphi / k

    Es *= nmp.exp(1.0j*kr)
    
    return Es
