import numpy as nmp
from lorenzmie import lm_angular_spectrum
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

def discretize_plan(NA, M, nm_img):
    '''
    Discretizes a plane according to Eq. 130 - 131.
    '''
    nx = ny = 20 # (FIX ME: use criteria Eq 1.42)

    x = nmp.tile(nmp.arange(nx, dtype = float), ny)
    y = nmp.repeat(nmp.arange(ny, dtype = float), nx)

    sx = NA/(M*nm_img)*( (1 + 2*x)/(nx -1) - 1)
    sy = NA/(M*nm_img)*( (1 + 2*y)/(ny -1) - 1)

    return nx, ny, sx, sy
    

def consv_energy(es, sx_obj, sx_img):
    '''
    Changes electric field strength factor density to obey the conversation
    of energy.
    '''
    # Compute the necessary cosine terms in Eq 108.
    cos_theta_img = nmp.sqrt(1. - sx_img**2)
    cos_theta_obj = nmp.sqrt(1. - sx_obj**2)
    es *= nmp.sqrt(cos_theta_img/cos_theta_obj)

    return es
    

def debyewolf(x, y, z, ap, np, nm, 
              lamb = 0.447, mpp = 0.135, dim = [201,201],
              NA = 1.45, nm_obj = 1.5, nm_img = 1.0, 
              M = 100, f = 20.*10**5):
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

    # Necessary constants.
    ci = complex(0., 1.)
    k_img = 2*np.pi/lamb/nm_img

    # Compute grids sx_obj and sx_img.
    p, q, sx_img, sy_img = discretize_plane(NA, M, nm_img)
    np, nq = 2*p, 2*q
    npts = p*q
    sx_obj = M*nm_img/nm_obj*sx
    sy_obj = M*nm_img/nm_obj*sy
    
    # Compute the electromagnetic strength factor on the object side (Eq 40 Ref[1]).
    ab = sphere_coefficients(ap,np,lamb,mpp)
    es_obj = lm_angular_spectrum(sx, sy, ab, lamb, nm_obj, f/M)

    # Displace the field.
    #es *= phase_displace(x, y, z, r, k)

    # Ensure conservation of energy is observed with abbe sine condition.
    es_img = consv_energy(es_obj)

    # Compute auxiliary (Eq. 133) with zero padding!
    aber  = nmp.zeros([npts, 3], complex) # As a function of sx_img, sy_img
    g_aux = nmp.zeros([npts, 3], complex)
    g_aux[p/2:3*p/2, q/2:3*q/2] = es_img[:]/np.sqrt(1. - sx_img**2)*np.exp(-ci*k_img*aber)
    g_aux = g_aux.reshape(np, nq, 3)

    # Apply discrete Fourier Transform (Eq. 135).
    es_m_n  = nmp.fft.fft2(g_aux)

    # Compute the electric field at the imaging plane
    es_cam  = (ci*NA**2/(M**2*nm_img*lamb))*(4/npts)*e_m_n
    m = np.arange(0,np)
    n = np.arange(0,nq)
    es_cam *= np.exp(-2*np.pi*ci*( (1-p)/(p**2*np)*m + (1-q)/(q**2*nq)*n)

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
    path_len = f + f/M
    field *= nmp.exp(nmp.complex(0.0, -k*path_len))
    field[:,0] += 1.0 # Plane wave normalized to amplitude 1.
    
    image = nmp.sum(nmp.real(field*conj(field)), axis = 1)

    return image.reshape(nx,ny)
