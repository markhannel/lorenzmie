import numpy as nmp
from sphere_coefficients import sphere_coefficients

def phase_displace(x, y, z, r, k):
    ''' Determines the phase due to displacement. '''
    # Compute R
    R = r**2+2*z*(nmp.sqrt(r**2-(x**2+y**2)))+z**2 # R squared.
    R = nmp.sqrt(R)
    
    phase = nmp.exp(nmp.complex(0.,i*k*(R-r)))
    return phase

def check_if_numpy(x, char_x):
    ''' checks if x is a numpy array '''
    if type(x) != nmp.ndarray:
        print char_x + ' must be an numpy array'
        return False
    else:
        return True

def sphericalfield(x, y, f, ab, lamb, cartesian = False):
    """
    Calculate the complex electric field defined by an array of scattering
    coefficients. 

    Args:
        x: [npts] array of pixel coordinates [pixels]
        y: [npts] array of pixel coordinates [pixels]
        f: [float] distance to focal plane [pixels]
        ab: [2,nc] array of a and b scattering coefficients, where
            nc is the number of terms required for convergence.
    Keywords:
        lamb: wavelength of light in medium [pixels]
        cartesian: if True, field is expressed as (x,y,z) else (r, theta, phi)
    Returns:
        field: [3,npts] scattered electricc field
    """

    # Check that inputs are numpy arrays
    for var, char_var in zip([x,y,ab], ['x', 'y', 'ab']):
        if check_if_numpy(var, char_var) == False:
            print 'x, y and ab must be numpy arrays'
            return None

    if x.shape != y.shape:
        print 'x has shape {} while y has shape {}'.format(x.shape, y.shape)
        print 'and yet their dimensions must match.'
        return None

    npts = len(x)
    nc = len(ab[:,0])-1     # number of terms required for convergence

    k = 2.0 * nmp.pi / lamb # wavenumber in medium [pixel^-1]

    ci = complex(0,1.0)

    # Compute relevant coordinates systems.
    r = f
    rho   = nmp.sqrt(x**2 + y**2)
    z     = nmp.sqrt(r**2-rho**2)
    theta = nmp.arctan(rho/z)
    phi   = nmp.arctan(y/x)
    costheta = nmp.cos(theta)
    sintheta = nmp.sin(theta)
    cosphi   = nmp.cos(phi)
    sinphi   = nmp.sin(phi)
    
    #Fix arctan results of NAN to zero.
    b = nmp.where(theta ==0)[0]
    theta[b] = 0
    b = nmp.where(phi == 0)[0]
    phi[0] = 0

    kr = k*r  # reduced radial coordinate

    # starting points for recursive function evaluation ...
    # ... Riccati-Bessel radial functions, page 478
    sinkr = nmp.sin(kr)
    coskr = nmp.cos(kr)

    # FIXME (MDH): What about particles below the focal plane?
    xi_nm2 = coskr + ci*sinkr # \xi_{-1}(kr) 
    xi_nm1 = sinkr - ci*coskr # \xi_0(kr)    

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
    if cartesian == True:
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


def debyewolf(x, y, z, ap, np, nm, lamb = 0.447, mpp = 0.135, dim = [201,201]):
    '''
    Returns the electric fields in the imaging plane due to a spherical
    scatterer at (x,y,z) with radius ap and refractice index np.

    Args:
        x,y:   [pix] define where the image will be evaluated.
        z:     [um] scatterer's distance from the focal plane.
        ap:    [um] sets the radius of the spherical scatterer.
        np:    [R.I.U.] sets the refractive index of the scatterer.
        nm:    [R.I.U.] sets the refractive index of the medium.
        lamb:  [um] sets the wavelength of the coherent illumination.
               Default: 0.447 um. 
        mpp:   [um/pix] sets the size of a pixel.
               Default: 0.135.

    Return:
        field: [3,npts]

    Ref[1]: Capoglu et al. (2012). "The Microscope in a Computer:...", 
            Applied Optics, 38(34), 7085.
    '''

    # Check inputs
    
    # Necessary Physics constants.
    npts = len(x)
    nair = 1.00
    M = 100
    k = 2.0*nmp.pi/(lamb/nm/mpp)       # [pix^-1] wavelength in medium.
    k_air = 2.0*nmp.pi/(lamb/nair/mpp) # [pix^-1] wavelength in air.

    # Compute grid of points necessary for computation.
    nx, ny = map(float, dim)
    x = nmp.tile(nmp.arange(nx), ny)
    y = nmp.repeat(nmp.arange(ny), nx)
    x -= nx/2
    y -= ny/2.
    x,y = map(lambda x:x/mpp, (x,y))

    
    # Compute spherical coords of given grid.
    rho   = nmp.sqrt(x**2 + y**2)
    r     = nmp.sqrt(rho**2 + z**2)
    theta = nmp.arctan(rho/z)
    phi   = nmp.arctan(y/x)
    costheta = nmp.cos(theta)
    sintheta = nmp.sin(theta)
    cosphi   = nmp.cos(phi)
    sinphi   = nmp.sin(phi)

    # Compute the electromagnetic strength factor (Eq 40 Ref[1])
    ab = sphere_coefficients(ap,np,lamb,mpp)
    Es = spherefield(x, y, z, ab, lamb, cartesian = False)
    Es *= r*nmp.exp(nmp.complex(0.0, k*r))

    # Displace the field.
    Es *= phase_displace(x, y, z, r, k)

    ### Implement k-vector refraction and angular demagnification
    '''
    Calculate the polar angles of the scattered field arriving
    at the camera. Assumes the camera is in air.. Note for 
    lenless holography the camera may be potted with PDMS
    '''
    sintheta_p = nm*sintheta/(M*nair) # Abbe Sine Condition
    costheta_p = 1-sintheta_p**2
 
    '''
    Modulate the intensity such that energy is conserved between
    the entrance and exit pupil of the objective accounting for the
    difference between the solid angles entering and exiting the
    objective.
    '''
    factor = nmp.sqrt(costheta_p/costheta)
    Es[1,*] *= factor
    Es[2,*] *= factor
    
    # Rotate the k-vector from direction (theta, phi) to (theta_p, phi)
    sintheta = sintheta_p
    costheta = costheta_p

    # FFT
    E_img = nmp.zeros([npts,3],complex)
    # Scale the spatial frequencies and cosine term such that the
    # Exponential arg of the debye wolf integral matches the numpy.fft
    # definition.
    costheta 
    fft_arg  = nmp.complex(0.,1.0)*k_air/(2*nmp.pi)/costheta*E_img
    E_img  = nmp.fft.fft2(fft_arg)

    # Convert E_img to cartesian coords
    field = nmp.zeros([npts,3],complex)
    field += E_img
    
    field[:,0] =  E_img[:,0] * sintheta * cosphi
    field[:,0] += E_img[:,1] * costheta * cosphi
    field[:,0] -= E_img[:,2] * sinphi
    
    field[:,1] =  E_img[:,0] * sintheta * sinphi
    field[:,1] += E_img[:,1] * costheta * sinphi
    field[:,1] += E_img[:,2] * cosphi

    field[:,2] =  E_img[:,0] * costheta - E_img[:,1] * sintheta

    # Recombine with plane wave.
    path_len = z+r+2*Z # (FIXME (MDH): What should the path length be?)
    field *= nmp.exp(nmp.complex(0.0, -k*path_len))
    field[:,0] += 1.0 # Plane wave normalized to amplitude 1.
    
    image = nmp.sum(nmp.real(field*conj(field)), axis = 1)

    return image.reshape(nx,ny)
