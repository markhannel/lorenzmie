import numpy as np
import matplotlib.pyplot as plt

def check_if_numpy(x, char_x):
    ''' checks if x is a numpy array '''
    if type(x) != np.ndarray:
        print char_x + ' must be an numpy array'
        return False
    else:
        return True


def lm_angular_spectrum(sx, sy, ab, lamb, n_m, f, z = 0):
    """
    Calculate the angular spectrum of the scattered field of a lorenz-mie
    scatterer with sphere coefficients ab. Result is computer in vector
    spherical coordinates.

    Args:
        sx: [npts] array of pixel coordinates.
        sy: [npts] array of pixel coordinates.
        ab: [2,nc] array of a and b scattering coefficients, where
            nc is the number of terms required for convergence.
        lamb: [um] sets the wavelength of the scattered light.
        n_m: [R.I.U.] refractive index of the medium immersing the scatterer.
        f:  [float] distance to focal plane
        z:  [um] distance from the focal plane to the scatterer.
            Default: 0 um

    Returns:
        ang_spec: [3,npts] angular spectrum in spherical coordinates.
    """

    # Necessary constants.
    npts = len(sx)
    nc = len(ab[:,0])-1     # number of terms required for convergence

    k = 2.0*np.pi*n_m/ lamb # wavenumber in medium [pixel^-1]
    
    # Compute the points at which we want to compute the electric field.
    '''
     We want to compute the field on a spherical surface $S_1$ centered
     at the focal plane. However, if z is not 0 um, the scatterer is not 
     centered in the focal plane. Find the points (r,\theta,\phi) which
     define the distances between the scatterer and the desired computation
     points on the spherical surface $S_1$.
    '''
    rho_sq = sx**2+sy**2
    phi = np.arctan2(sy, sx)
    theta = np.arctan2(np.sqrt(rho_sq), z + f)
    sintheta = np.sin(theta)
    costheta = np.cos(theta)
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    r = (z+f)**2/np.sqrt((z+f)**2+rho_sq)

    kr = k*r
    kf = k*f

    # starting points for recursive function evaluation ...
    # ... Riccati-Bessel radial functions, page 478
    sinkr = np.sin(kr)
    coskr = np.cos(kr)

    # FIXME (MDH): What about particles below the focal plane?
    xi_nm2 = coskr + 1.0j*sinkr # \xi_{-1}(kr) 
    xi_nm1 = sinkr - 1.0j*coskr # \xi_0(kr)    

    # ... angular functions (4.47), page 95
    pi_nm1 = 0.0                    # \pi_0(\cos\theta)
    pi_n   = 1.0                    # \pi_1(\cos\theta)
 
    # storage for vector spherical harmonics: [r,theta,phi]
    Mo1n = np.zeros([3,npts],complex)
    Ne1n = np.zeros([3,npts],complex)

    # storage for scattered field
    Es = np.zeros([3,npts],complex)
        
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
    Es[0,:] *= cosphi * sintheta / (k**2*f)
    Es[1,:] *= cosphi / k  # Didn't divide by r. See Eq 40 Ref 1.
    Es[2,:] *= sinphi / k  

    # Complete Eq. 40 Ref 1.
    Es *= np.exp(1.0j*kf)
    
    return Es


def test_lm_ang_spectrum():
    import geometry as g
    from sphere_coefficients import sphere_coefficients
    import matplotlib.pyplot as plt

    z = 10
    a_p = 0.5
    n_p = 1.5
    nm_obj = 1.339
    nm_img = 1.0
    NA = 1.45
    lamb = 0.447
    mpp = 0.135
    M = 100
    f = 2.*10**5
    dim = [201,201]
    r = f/M
    p, q = 250, 250
    origin = [0.5*(p-1.), 0.5*(q-1.)]
    obj_factor = 2*NA/nm_obj
    obj_scale = [obj_factor*1./p, obj_factor*1./q]

    s_obj_cart = g.CartesianCoordinates(p, q, origin, obj_scale)
    
    ab = sphere_coefficients(a_p, n_p, nm_obj, lamb)
    sx = s_obj_cart.xx.ravel()
    sy = s_obj_cart.yy.ravel()

    inds = np.where(sx**2+sy**2 < (NA/nm_obj)**2)[0]

    sx *= r*NA/nm_obj
    sy *= r*NA/nm_obj

    ang_spec = np.zeros([3, p*q], dtype = complex)
    ang_spec[:, inds] = lm_angular_spectrum(sx[inds], sy[inds], ab, lamb, nm_obj, f/M, z)
    ang_spec = ang_spec.reshape(3, p, q)

    # Auxiliary Field at P2.
    plt.plot(np.abs(ang_spec[1,:]))
    plt.show()
    plt.imshow(np.hstack([np.abs(ang_spec[0]), np.abs(ang_spec[1]), np.abs(ang_spec[2])]))
    plt.title(r'ang_spec before $(r, \theta, \phi)$ at $P_2$') 
    plt.show()

def test_against_old():
    '''Produces an image using the angular spectrum and using the old method of image formation.'''
    pass

def test_r_dependence():
    '''Tests that the angular spectrum doesnt depend greatly on r.'''
    pass


if __name__ == '__main__':
    test_lm_ang_spectrum()
