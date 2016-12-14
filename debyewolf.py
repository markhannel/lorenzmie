import numpy as nmp
from lorenzmie import lm_angular_spectrum
from sphere_coefficients import sphere_coefficients
import matplotlib.pyplot as plt
import matplotlib as mp
mp.rcParams.update({'font.size':22})


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

def aperture(field, x, y, r_max):
    '''Sets field to zero wherever x**2+y**2 >= rmax.'''
    r_2 = x**2+y**2
    indices = nmp.where(r_2 >= r_max**2)

    field[:,indices] = 0
    return field

def discretize_plan(NA, M, lamb, nm_img, mpp):
    '''
    Discretizes a plane according to Eq. 130 - 131.
    '''
    # Suppose the largest scatterer we consider is 20 lambda. THen
    # P should be larger than 40*NA.
    diam = 200 # wavelengths
    p, q = int(diam*NA), int(diam*NA)

    # Pad with zeros for increased resolution and to set del_x to mpp.
    pad_p = int((lamb - mpp*2*NA)/(mpp*2*NA)*p)
    pad_q = int((lamb - mpp*2*NA)/(mpp*2*NA)*q)

    return pad_p, pad_q, p, q
    

def consv_energy(es, sx_obj, sy_obj, sx_img, sy_img, r_max):
    '''
    Changes electric field strength factor density to obey the conversation
    of energy.
    '''
    r_2 = sx_obj**2 +sy_obj**2
    indices = nmp.where(r_2 <= r_max**2)
    # Compute the necessary cosine terms in Eq 108.
    cos_theta_img = nmp.sqrt(1. - (sx_img**2+sy_img**2))
    cos_theta_obj = nmp.sqrt(1. - (sx_obj**2+sy_obj**2))
    es[:,indices[0],indices[1]] *= nmp.sqrt(cos_theta_img[indices[0],indices[1]]/cos_theta_obj[indices[0],indices[1]])

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
    pad_p, pad_q, p, q = discretize_plan(NA, M, lamb, nm_img, mpp)
    pad_p = 0
    pad_q = 0 
    Np = pad_p + p
    Nq = pad_q + q

    npts = p*q


    x = nmp.tile(nmp.arange(p, dtype = float), q)
    y = nmp.repeat(nmp.arange(q, dtype = float), p)

    sx_img = NA/(M*nm_img)*( (1+2*x)/p - 1)
    sy_img = NA/(M*nm_img)*( (1+2*y)/q - 1)


    sx_obj = M*nm_img/nm_obj*sx_img
    sy_obj = M*nm_img/nm_obj*sy_img

    # Compute the angular spectrum incident on plane 1.
    # Compute the electromagnetic strength factor on the object side (Eq 40 Ref[1]).
    ab = sphere_coefficients(ap,np,lamb,mpp)
    es_obj = lm_angular_spectrum(sx_obj, sy_obj, ab, lamb, nm_obj, f/M)
    es_obj = nmp.nan_to_num(es_obj)

    temp = es_obj.reshape(3,p,q)
    plt.imshow(nmp.hstack([nmp.abs(temp[0]), nmp.abs(temp[1]), nmp.abs(temp[2])]))
    plt.title(r'Electric Field strength $(r, \theta, \phi)$ at $P_1$')
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()


    # Apply the aperture function.
    es_obj = aperture(es_obj, sx_obj, sy_obj, NA/nm_obj)
    es_obj = es_obj.reshape(3,p,q)


    plt.imshow(nmp.hstack([nmp.abs(es_obj[0]), nmp.abs(es_obj[1]), nmp.abs(es_obj[2])]))
    plt.title(r'Electric Field strength  $(r, \theta, \phi)$ at $P_1$ after aperture')
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()

    plt.show()

    sx_img = nmp.arange(p, dtype = float)
    sy_img = nmp.arange(q, dtype = float)
    
    sx_img = NA/(M*nm_img)*( (1+2*sx_img)/p - 1)
    sy_img = NA/(M*nm_img)*( (1+2*sy_img)/q - 1)
    
    # Displace the field.
    #es *= phase_displace(x, y, z, r, k)

    sxx_img, syy_img = nmp.meshgrid(sx_img, sy_img)
    sintheta = sxx_img**2 + syy_img**2
    costheta = nmp.sqrt(1. - sintheta)
    sintheta = nmp.sqrt(sintheta)

    sxx_obj = M*nm_img/nm_obj*sxx_img
    syy_obj = M*nm_img/nm_obj*syy_img

    # Compute the electric field strength factor on plane 2.
    # Ensure conservation of energy is observed with abbe sine condition.
    es_img = consv_energy(es_obj, sxx_obj, syy_obj, sxx_img, syy_img, NA/nm_obj)
    es_img = remove_r(es_img) # Should be no r component.

    es_img = nmp.nan_to_num(es_img)
    

    plt.imshow(nmp.hstack([nmp.abs(es_img[0]), nmp.abs(es_img[1]), nmp.abs(es_img[2])]))
    plt.title(r'Electric Field strength $(r, \theta, \phi)$ at $P_2$')
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()

    plt.show()


    # Compute auxiliary (Eq. 133) with zero padding!
    #aber  = nmp.zeros([3, Np, Nq], complex) # As a function of sx_img, sy_img
    g_aux = nmp.zeros([3, Np, Nq], complex)
    for i in xrange(3):
        #g_aux[i, pad_p/2:-pad_p/2, pad_q/2:-pad_q/2] = es_img[i,:,:]/costheta
        g_aux[i, :,:] = es_img[i,:,:]/costheta
        #g_aux *= nmp.exp(-1.j*k_img*aber)

    plt.imshow(nmp.hstack([nmp.abs(g_aux[0]), nmp.abs(g_aux[1]), nmp.abs(g_aux[2])]))
    plt.title(r'Auxiliary field $(r, \theta, \phi)$ at $P_2$')
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()

    plt.show()


    # Apply discrete Fourier Transform (Eq. 135).
    es_m_n = nmp.fft.fft2(g_aux, s = (Np,Nq))
    for i in xrange(3):
        es_m_n[i] = nmp.fft.fftshift(es_m_n[i])

    plt.imshow(nmp.hstack([nmp.abs(es_m_n[0]), nmp.abs(es_m_n[1]), nmp.abs(es_m_n[2])]))
    plt.title(r'Fourier Transform of aux $(r, \theta, \phi)$')
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()

    plt.show()


    # Compute the electric field at plane 3.
    # Accounting for aliasing.
    es_cam  = (1.j*NA**2/(M*lamb))*(4./npts)*es_m_n

    m = nmp.arange(0,Np, dtype = float)
    n = nmp.arange(0,Nq, dtype = float)
    mm, nn = nmp.meshgrid(m,n)
    for i in xrange(3):
        es_cam[i,:,:] *= nmp.exp(-1.j*nmp.pi*( mm*(1.-p)/Np + nn*(1.-q)/Nq))

    plt.imshow(nmp.hstack([nmp.abs(es_cam[0]), nmp.abs(es_cam[1]), nmp.abs(es_cam[2])]))
    plt.title(r'Electric field $(r, \theta, \phi)$ at the camera plane after dealiasing')
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()

    nx_img = nmp.arange(Np, dtype = float)
    ny_img = nmp.arange(Nq, dtype = float)

    nx_img = NA/(M*nm_img)*((1+2*nx_img-pad_p)/p-1)
    ny_img = NA/(M*nm_img)*((1+2*ny_img-pad_q)/q-1)

    nxx_img, nyy_img = nmp.meshgrid(nx_img, ny_img)
    sintheta = nxx_img**2 + nyy_img**2

    costheta = nmp.sqrt(1 - sintheta**2)
    sintheta = nmp.sqrt(sintheta)

    cosphi = nxx_img/sintheta
    sinphi = nyy_img/sintheta

    # Convert E_img to cartesian coords
    es_cam_cart = nmp.zeros([3,Np,Nq], dtype = complex)
    es_cam_cart += es_cam

    es_cam_cart[0,:,:] =  es_cam[0,:,:] * sintheta * cosphi
    es_cam_cart[0,:,:] += es_cam[1,:,:] * costheta * cosphi
    es_cam_cart[0,:,:] -= es_cam[2,:,:] * sinphi

    es_cam_cart[1,:,:] =  es_cam[0,:,:] * sintheta * sinphi
    es_cam_cart[1,:,:] += es_cam[1,:,:] * costheta * sinphi
    es_cam_cart[1,:,:] += es_cam[2,:,:] * cosphi

    es_cam_cart[2,:,:] =  es_cam[0,:,:] * costheta - es_cam[1,:,:] * sintheta

    
    plt.imshow(nmp.hstack([nmp.abs(es_cam_cart[0]), nmp.abs(es_cam_cart[1]), nmp.abs(es_cam_cart[2])]))
    plt.title(r'Electric field $(x,y,z)$ at the camera plane after dealiasing')
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()

    plt.show()

    # Recombine with plane wave.
    #path_len = f + f/M
    path_len = 0.2/k_img
    es_cam_cart[0,:,:] += 1.0*nmp.exp(-1.j*k_img*path_len) # Plane wave normalized to amplitude 1.
    
    image = nmp.sum(nmp.real(es_cam_cart*nmp.conjugate(es_cam_cart)), axis = 0)
    '''
    real_part = nmp.sum(nmp.real(es_cam_cart), axis = 0)
    imag_part = nmp.sum(nmp.imag(es_cam_cart), axis = 0)

    print nmp.mean(image)
    plt.imshow(nmp.hstack([real_part,imag_part]))
    plt.title('Es_cam_cart real and imaginary')
    plt.show()
    '''
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
    ap = 2.0
    np = 1.5

    image = debyewolf(z, ap, np, lamb = 0.447, mpp = 0.135, dim = [201,201], NA = 1.45, 
                      nm_obj = 1.339, nm_img = 1.0, M = 100, f = 20.*10**5)

    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.title('Final Image')
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.gray()
    plt.show()

    

if __name__ == '__main__':
    test_debye()
