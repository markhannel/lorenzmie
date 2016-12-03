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

    # Pad with zeros for increased resolution and to set del_x to mpp.
    pad_p = int((lamb - mpp*2*NA)/(mpp*2*NA)*p)
    pad_q = int((lamb - mpp*2*NA)/(mpp*2*NA)*q)

    x = nmp.tile(nmp.arange(p, dtype = float), q)
    y = nmp.repeat(nmp.arange(q, dtype = float), p)

    sx_img = NA/(M*nm_img)*( (1+2*x)/p - 1)
    sy_img = NA/(M*nm_img)*( (1+2*y)/q - 1)

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

def remove_r(es):
    '''Remove r component of vector.'''
    es[0,:,:] = 0.0
    return es
    

def debyewolf(z, ap, np, lamb = 0.447, mpp = 0.135, dim = [201,201], NA = 1.45, 
              nm_obj = 1.339, nm_img = 1.0, M = 100, f = 20.*10**5):
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
    k_img = 2*nmp.pi*nm_img/lamb

    # Compute grids sx_obj and sx_img.
    pad_p, pad_q, p, q, sx_img, sy_img = discretize_plan(NA, M, lamb, nm_img, mpp)
    Np = pad_p + p
    Nq = pad_q + q

    npts = p*q

    sx_obj = M*nm_img/nm_obj*sx_img
    sy_obj = M*nm_img/nm_obj*sy_img
    
    print max(sx_img**2+sy_img**2)
    print max(sx_obj**2+sy_obj**2)

    # Compute the electromagnetic strength factor on the object side (Eq 40 Ref[1]).
    ab = sphere_coefficients(ap,np,lamb,mpp)
    es_obj = lm_angular_spectrum(sx_obj, sy_obj, ab, lamb, nm_obj, f/M)
    es_obj = nmp.nan_to_num(es_obj)
    es_obj = es_obj.reshape(3,p,q)
    
    import matplotlib.pyplot as plt
    plt.imshow(nmp.real(es_obj[0,:,:]))
    plt.show()

    sx_img = nmp.arange(p, dtype = float)
    sy_img = nmp.arange(q, dtype = float)
    
    sx_img = NA/(M*nm_img)*( (1+2*sx_img)/(p-1) - 1)
    sy_img = NA/(M*nm_img)*( (1+2*sy_img)/(q-1) - 1)
    
    sx_obj = M*nm_img/nm_obj*sx_img
    sy_obj = M*nm_img/nm_obj*sy_img

    # Displace the field.
    #es *= phase_displace(x, y, z, r, k)

    # Ensure conservation of energy is observed with abbe sine condition.
    es_img = consv_energy(es_obj, sx_obj, sy_obj, sx_img, sy_img)
    es_img = remove_r(es_img) # Should be no r component.

    # Compute auxiliary (Eq. 133) with zero padding!
    # (FIXME MDH: Set up a 2d version of sx_img, sy_img with meshgrid)
    #aber  = nmp.zeros([3, Np, Nq], complex) # As a function of sx_img, sy_img
    sxx_img, syy_img = nmp.meshgrid(sx_img, sy_img)
    sintheta = sxx_img**2 + syy_img**2
    costheta = nmp.sqrt(1 - sintheta)
    sintheta = nmp.sqrt(sintheta)
    
    g_aux = nmp.zeros([3, p, q], complex)
    for i in xrange(3):
        g_aux[i,:,:] = es_img[i,:,:]/costheta
        #g_aux *= nmp.exp(-1.j*k_img*aber)

    # Apply discrete Fourier Transform (Eq. 135).
    es_m_n = nmp.fft.fft2(g_aux, s = (Np, Nq))
    es_m_n = nmp.fft.fftshift(es_m_n)

    # Compute the electric field at the image while accounting for aliasing.
    es_cam  = (1.j*NA**2/(M**2*nm_img*lamb))*(4/npts)*es_m_n
    m = nmp.arange(0,Np)
    n = nmp.arange(0,Nq)
    mm, nn = nmp.meshgrid(m,n)
    for i in xrange(3):
        es_cam[i,:,:] *= nmp.exp(-2*nmp.pi*1.j*( (1-p)/(p**2*Np*mm + (1-q)/(q**2*Nq)*nn)))

    sx_img = nmp.arange(Np, dtype = float)
    sy_img = nmp.arange(Nq, dtype = float)
    
    sx_img = NA/(M*nm_img)*( (1+2*sx_img)/(p-1) - 1)
    sy_img = NA/(M*nm_img)*( (1+2*sy_img)/(q-1) - 1)

    sxx_img, syy_img = nmp.meshgrid(sx_img, sy_img)
    sintheta = sxx_img**2 + syy_img**2
    costheta = nmp.sqrt(1 - sintheta**2)

    cosphi = sxx_img/sintheta
    sinphi = syy_img/sintheta

    # Convert E_img to cartesian coords
    es_cam_cart = nmp.zeros([3, Np, Nq], complex)
    es_cam_cart += es_cam 

    es_cam_cart[0,:,:] =  es_cam[0,:,:] * sintheta * cosphi
    es_cam_cart[0,:,:] += es_cam[1,:,:] * costheta * cosphi
    es_cam_cart[0,:,:] -= es_cam[2,:,:] * sinphi

    es_cam_cart[1,:,:] =  es_cam[0,:,:] * sintheta * sinphi
    es_cam_cart[1,:,:] += es_cam[1,:,:] * costheta * sinphi
    es_cam_cart[1,:,:] += es_cam[2,:,:] * cosphi

    es_cam_cart[2,:,:] =  es_cam[0,:,:] * costheta - es_cam[1,:,:] * sintheta
    
    # Recombine with plane wave.
    path_len = f + f/M
    es_cam_cart[0,:,:] += 1.0*nmp.exp(-1.j*k_img*path_len) # Plane wave normalized to amplitude 1.
    
    image = nmp.sum(nmp.real(es_cam_cart*nmp.conj(es_cam_cart)), axis = 0)

    return image

def test_discretize():
    NA = 1.45
    M = 100
    lamb = 0.447
    nm_img = 1.0
    mpp = 0.135
    pad_p, pad_q, p, q, sx, sy = discretize_plan(NA, M, lamb, nm_img, mpp)
    
    del_x = lamb*p*M/(2*NA*(pad_p+p))
    print del_x/M

def test_debye():    
    z = 0.0
    ap = 0.5
    np = 1.5

    image = debyewolf(z, ap, np, lamb = 0.447, mpp = 0.135, dim = [201,201], NA = 1.45, 
                      nm_obj = 1.339, nm_img = 1.0, M = 100, f = 20.*10**5)

    print image[0:10]
    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.gray()
    plt.show()

    

if __name__ == '__main__':
    test_debye()
