import numpy as nmp
from lorenzmie import lm_angular_spectrum
from sphere_coefficients import sphere_coefficients

def phase_displace(x, y, z, r, k):
    ''' Determines the phase due to displacement. '''
    # Compute R
    R = r**2+2*z*(nmp.sqrt(r**2-(x**2+y**2)))+z**2 # R squared.
    R = nmp.sqrt(R)
    phase = nmp.exp(nmp.complex(0.,1j*k*(R-r)))

    return phase

def check_if_numpy(x, char_x):
    ''' checks if x is a numpy array '''
    if type(x) != nmp.ndarray:
        print char_x + ' must be an numpy array'
        return False
    else:
        return True

def discretize_plan(NA, M, lamb, nm_img, mpp):
    '''
    Discretizes a plane according to Eq. 130 - 131.
    '''
    # Suppose the largest scatterer we consider is 20 lambda. THen
    # P should be larger than 40*NA.
    p, q = int(40*NA), int(40*NA)


    print mpp, NA
    # Pad with zeros for increased resolution and to set del_x to mpp.
    pad_p = int((lamb - mpp*2*NA)/(mpp*2*NA)*p)
    pad_q = int((lamb - mpp*2*NA)/(mpp*2*NA)*q)

    x = nmp.tile(nmp.arange(p, dtype = float), q)
    y = nmp.repeat(nmp.arange(q, dtype = float), p)

    sx_img = NA/(M*nm_img)*( (1+2*x)/(p-1) - 1)
    sy_img = NA/(M*nm_img)*( (1+2*y)/(q-1) - 1)

    return pad_p, pad_q, p, q, sx_img, sy_img
    

def consv_energy(es, sx_obj, sy_obj, sx_img, sy_img):
    '''
    Changes electric field strength factor density to obey the conversation
    of energy.
    '''
    # Compute the necessary cosine terms in Eq 108.
    cos_theta_img = nmp.sqrt(1. - (sx_img**2+sy_img**2))
    cos_theta_obj = nmp.sqrt(1. - (sx_obj**2+sy_obj**2))
    es *= nmp.sqrt(cos_theta_img/cos_theta_obj)

    return es
    

def debyewolf(z, ap, np, nm, lamb = 0.447, mpp = 0.135, dim = [201,201], NA = 1.45, 
              nm_obj = 1.5, nm_img = 1.0, M = 100, f = 20.*10**5):
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
    k_img = 2*np.pi*nm_img/lamb

    # Compute grids sx_obj and sx_img.
    pad_p, pad_q, p, q, sx_img, sy_img = discretize_plan(NA, M, nm_img)
    Np = pad_p + p
    Nq = pad_q + q

    npts = p*q
    sx_obj = M*nm_img/nm_obj*sx
    sy_obj = M*nm_img/nm_obj*sy
    
    # Compute the electromagnetic strength factor on the object side (Eq 40 Ref[1]).
    ab = sphere_coefficients(ap,np,lamb,mpp)
    es_obj = lm_angular_spectrum(sx_obj, sy_obj, ab, lamb, nm_obj, f/M)
    es_obj = es_obj.reshape(3,p,q)

    # Displace the field.
    #es *= phase_displace(x, y, z, r, k)

    # Ensure conservation of energy is observed with abbe sine condition.
    es_img = consv_energy(es_obj)

    # Compute auxiliary (Eq. 133) with zero padding!
    aber  = nmp.zeros([3, npts], complex) # As a function of sx_img, sy_img
    g_aux = nmp.zeros([3, npts], complex)
    g_aux[:,pad_p/2:-pad_p/2, pad_q/2:-pad_q/2] = es_img/np.sqrt(1. - sx_img**2)*np.exp(-1.j*k_img*aber)
    g_aux = g_aux.reshape(3, Np, Nq)

    # Apply discrete Fourier Transform (Eq. 135).
    es_m_n  = nmp.fft.fft2(g_aux)

    # Compute the electric field at the imaging plane
    es_cam  = (1.j*NA**2/(M**2*nm_img*lamb))*(4/npts)*es_m_n
    m = np.arange(0,Np)
    n = np.arange(0,Nq)
    es_cam *= np.exp(-2*np.pi*1.j*( (1-p)/(p**2*Np*m + (1-q)/(q**2*Nq)*n)))

    # Convert E_img to cartesian coords
    field = nmp.zeros([3, npts],complex)
    field += es_cam # FIXME!!
    
    # Recombine with plane wave.
    path_len = f + f/M
    field *= nmp.exp(-1.j*k_img*path_len)
    field[:,0] += 1.0 # Plane wave normalized to amplitude 1.
    
    image = nmp.sum(nmp.real(field*nmp.conj(field)), axis = 1)

    return image.reshape(dim[0],dim[1])

if __name__ == '__main__':
    NA = 1.45
    M = 100
    lamb = 0.447
    nm_img = 1.0
    mpp = 0.135
    pad_p, pad_q, p, q, sx, sy = discretize_plan(NA, M, lamb, nm_img, mpp)
    
    del_x = lamb*p*M/(2*NA*(pad_p+p))
    print del_x/M
