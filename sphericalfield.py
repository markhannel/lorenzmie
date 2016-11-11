import numpy as nmp

def check_if_numpy(x, char_x):
    ''' checks if x is a numpy array '''
    if type(x) != nmp.ndarray:
        print char_x + ' must be an numpy array'
        return False
    else:
        return True

def sphericalfield(x, y, z, ab, lamb, cartesian = True):
    """
    Calculate the complex electric field defined by an array of scattering
    coefficients.

    Args:
        x: [npts] array of pixel coordinates [pixels]
        y: [npts] array of pixel coordinates [pixels]
        z: If field is required in a single plane, then
            z is the plane's distance from the sphere's center
            [pixels].
        a: [2,nc] array of a and b scattering coefficients, where
            nc is the number of terms required for convergence.
    Keywords:
        lamb: wavelength of light in medium [pixels]
        cartesian: if True, field is expressed as (x,yz) else (r, theta, phi)
    Returns:
        field: [3,npts] scattered electricc field
    """

    # Check that inputs are numpy arrays
    for var, char_var in zip([x,y,ab], ['x', 'y', 'ab']):
        if check_if_numpy(var, char_var) == False:
            print 'x, y and ab must be numpy arrays'
            return None

    if type(z) != int and type(z) != float:
        print 'z must be a float or int'
        return None

    # Check the inputs are the right size
    
    if x.shape != y.shape:
        print 'x has shape {} while y has shape {}'.format(x.shape, y.shape)
        print 'and yet their dimensions must match.'
        return None

    npts = len(x)
    nc = len(ab[:,0])-1     # number of terms required for convergence

    k = 2.0 * nmp.pi / lamb # wavenumber in medium [pixel^-1]

    ci = complex(0,1.0)

    # convert to spherical coordinates centered on the sphere.
    # (r, theta, phi) is the spherical coordinate of the pixel
    # at (x,y) in the imaging plane at distance z from the
    # center of the sphere.
    rho   = nmp.sqrt(x**2 + y**2)
    r     = nmp.sqrt(rho**2 + z**2)
    theta = nmp.arctan(rho/z)
    phi   = nmp.arctan(y/x)
    costheta = nmp.cos(theta)
    sintheta = nmp.sin(theta)
    cosphi   = nmp.cos(phi)
    sinphi   = nmp.sin(phi)
    
    #Fix arctan results of NAN to zero
    b = nmp.where(theta ==0)[0]
    theta[b] = 0
    b = nmp.where(phi == 0)[0]
    phi[0] = 0

    kr = k*r                        # reduced radial coordinate

    # starting points for recursive function evaluation ...
    # ... Riccati-Bessel radial functions, page 478
    sinkr = nmp.sin(kr)
    coskr = nmp.cos(kr)

    #xi_nm2 = coskr + nmp.sign(z)*ci*sinkr # \xi_{-1}(kr) 
    #xi_nm1 = sinkr - nmp.sign(z)*ci*coskr # \xi_0(kr)    
    xi_nm2 = coskr + ci*sinkr
    xi_nm1 = sinkr - ci*coskr
    # ... angular functions (4.47), page 95
    pi_nm1 = 0.0                    # \pi_0(\cos\theta)
    pi_n   = 1.0                    # \pi_1(\cos\theta)

    # storage for vector spherical harmonics: [r,theta,phi]
    Mo1n = nmp.zeros([npts,3],complex)
    Ne1n = nmp.zeros([npts,3],complex)

    # storage for scattered field
    Es = nmp.zeros([npts,3],complex)
        
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
        #Mo1n[:,0] = 0               # no radial component
        Mo1n[:,1] = pi_n * xi_n     # ... divided by cosphi/kr
        Mo1n[:,2] = tau_n * xi_n    # ... divided by sinphi/kr

        dn = (n * xi_n)/kr - xi_nm1
        Ne1n[:,0] = n*(n + 1.0) * pi_n * xi_n # ... divided by cosphi sintheta/kr^2
        Ne1n[:,1] = tau_n * dn      # ... divided by cosphi/kr
        Ne1n[:,2] = pi_n  * dn      # ... divided by sinphi/kr

        # prefactor, page 93
        En = ci**n * (2.0*n + 1.0) / n / (n + 1.0)

        # the scattered field in spherical coordinates (4.45)
        Es += En * (ci * ab[n,0] * Ne1n - ab[n,1] * Mo1n)
 
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
    Es[:,0] *= cosphi * sintheta / kr**2
    Es[:,1] *= cosphi / kr
    Es[:,2] *= sinphi / kr



    # By default, the scattered wave is returned in spherical
    # coordinates.  Project components onto Cartesian coordinates.
    # Assumes that the incident wave propagates along z and 
    # is linearly polarized along x
    if cartesian == 'cartesian':
        Ec = nmp.zeros([npts,3],complex)
        Ec += Es

        Ec[:,0] =  Es[:,0] * sintheta * cosphi
        Ec[:,0] += Es[:,1] * costheta * cosphi
        Ec[:,0] -= Es[:,2] * sinphi

        Ec[:,1] =  Es[:,0] * sintheta * sinphi
        Ec[:,1] += Es[:,1] * costheta * sinphi
        Ec[:,1] += Es[:,2] * cosphi


        Ec[:,2] =  Es[:,0] * costheta - Es[:,1] * sintheta

        return Ec
    else:
        return Es
