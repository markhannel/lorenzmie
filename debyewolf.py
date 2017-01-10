## FIXME: Determine if you should use fftshift or ifftshift or a 2d shift.

import numpy as np
from lorenzmie import lm_angular_spectrum
from sphere_coefficients import sphere_coefficients
import matplotlib.pyplot as plt
import geometry as g
from copy import deepcopy

def check_if_numpy(x, char_x):
    ''' checks if x is a numpy array '''
    if type(x) != np.ndarray:
        print char_x + ' must be an numpy array'
        return False
    else:
        return True

def aperture(field, x, y, r_max):
    '''Sets field to zero wherever x**2+y**2 >= rmax.'''
    r_2 = x**2+y**2
    indices = np.where(r_2 >= r_max**2)

    field[:,indices] = 0
    return field

def discretize_plan(NA, M, lamb, nm_img, mpp):
    '''Discretizes a plane according to Eq. 130 - 131.'''

    # Suppose the largest scatterer we consider is 20 lambda. Then
    # P should be larger than 40*NA.
    diam = 200 # wavelengths
    p, q = int(diam*NA), int(diam*NA)

    # Pad with zeros to help dealias and to set del_x to mpp.
    pad_p = int((lamb - mpp*2*NA)/(mpp*2*NA)*p)
    pad_q = int((lamb - mpp*2*NA)/(mpp*2*NA)*q)

    return pad_p, pad_q, p, q
    

def consv_energy(es, s_obj, s_img, r_max):
    '''
    Changes electric field strength factor density to obey the conversation
    of energy.
    '''
    sx_obj = s_obj.xx
    sy_obj = s_obj.yy
    sx_img = s_img.xx
    sy_img = s_img.yy
    r_2 = sx_obj**2 +sy_obj**2
    indices = np.where(r_2 <= r_max**2)

    # Compute the necessary cosine terms in Eq 108.
    cos_theta_img = np.sqrt(1. - (sx_img**2+sy_img**2))
    cos_theta_obj = np.sqrt(1. - (sx_obj**2+sy_obj**2))
    es[:,indices[0],indices[1]] *= np.sqrt(cos_theta_img[indices[0],indices[1]]/cos_theta_obj[indices[0],indices[1]])

    return es

def remove_r(es):
    '''Remove r component of vector.'''
    es[0,:,:] = 0.0
    return es
  
    
def scatter(s_obj_cart, a_p, n_p, nm_obj, lamb, r, z):
    '''Computes the scattered field.'''
    
    # Compute the electromagnetic strength factor on the object side 
    # (Eq 40 Ref[1]).
    ab = sphere_coefficients(a_p, n_p, nm_obj, lamb)
    es_obj = lm_angular_spectrum(s_obj_cart, ab, lamb, nm_obj, r, z)
    es_obj = np.nan_to_num(es_obj)

    # Apply the aperture function. (FIXME (MDH): Check isn't necessary)
    #es_obj = aperture(es_obj, s_obj_cart, NA/nm_obj)

    p, q = s_obj_cart.shape
    return es_obj.reshape(3,p,q)

def collection(es_obj, s_obj_cart, s_img_cart, nm_obj, NA):
    # Ensure conservation of energy is observed with abbe sine condition.
    es_img = consv_energy(deepcopy(es_obj), s_obj_cart, s_img_cart, NA/nm_obj)
    es_img = remove_r(es_img) # Should be no r component.

    es_img = np.nan_to_num(es_img)
    return es_img

def refocus(es_img, sph_img, n_disc_grid, p, q, Np, Nq, NA, M, lamb):
    '''Propagates the electric field from the exit pupil to the image plane.'''
    # Compute auxiliary (Eq. 133) with zero padding!
    #aber  = np.zeros([3, Np, Nq], complex) # As a function of sx_img, sy_img
    g_aux = np.zeros([3, p, q], complex)
    for i in xrange(3):
        g_aux[i, :,:] = es_img[i,:,:]/sph_img.costheta
        #g_aux *= np.exp(-1.j*k_img*aber)

    # Apply discrete Fourier Transform (Eq. 135).
    es_m_n = np.fft.fft2(g_aux, s = (Np,Nq))
    for i in xrange(3):
        es_m_n[i] = np.fft.fftshift(es_m_n[i])

    # Compute the electric field at plane 3.
    # Accounting for aliasing.
    es_cam  = (1.j*NA**2/(M*lamb))*(4./(p*q))*es_m_n 
    # FIXME (MDH): Should it be p*q or NpNq

    mm = n_disc_grid.xx
    nn = n_disc_grid.yy

    for i in xrange(3):
        es_cam[i,:,:] *= np.exp(-1.j*np.pi*( mm*(1.-p)/Np + nn*(1.-q)/Nq))

    return es_cam

def image_formation(es_cam, sph_n_img, k_img):
    '''Produces an image from the electric fields present.'''
    
    # Convert es_cam to cartesian coords
    es_cam_cart = g.spherical_to_cartesian(es_cam, sph_n_img)

    #path_len = f + f/M
    path_len = 0. # FIXME (MDH): What should the path length be?
    es_cam_cart[0,:,:] += 1.0*np.exp(-1.j*k_img*path_len) # Plane wave normalized to amplitude 1.
    
    image = np.sum(np.real(es_cam_cart*np.conjugate(es_cam_cart)), axis = 0)

    return image

def debyewolf(z, a_p, n_p, lamb = 0.447, mpp = 0.135, dim = [201,201], NA = 1.45, 
              nm_obj = 1.339, nm_img = 1.0, M = 100, f = 20.*10**5, quiet = True):
    '''
    Returns the electric fields in the imaging plane due to a spherical
    scatterer at (x,y,z) with radius a_p and refractice index n_p.

    Args:
        x,y:   [pix] define where the image will be evaluated.
        z:     [um] scatterer's distance from the focal plane.
        a_p:   [um] sets the radius of the spherical scatterer.
        n_p:   [R.I.U.] sets the refractive index of the scatterer.
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

    # Devise a discretization plan.
    pad_p, pad_q, p, q = discretize_plan(NA, M, lamb, nm_img, mpp)
    Np = pad_p + p
    Nq = pad_q + q

    # Compute the three geometries, s_img, s_obj, n_img
    # Origins for the coordinate systems.
    origin = [.5*(p-1.), .5*(q-1.)]
    
    # Scale factors.
    img_factor = 2*NA/(M*nm_img)
    obj_factor = 2*NA/nm_obj
    img_scale = [img_factor*1./p, img_factor*1./q]
    obj_scale = [obj_factor*1./p, obj_factor*1./q]

    # Cartesian Geometries.
    s_img_cart = g.CartesianCoordinates(p, q, origin, img_scale)
    s_obj_cart = g.CartesianCoordinates(p, q, origin, obj_scale)
    n_disc_grid = g.CartesianCoordinates(Np, Nq)
    n_img_cart = g.CartesianCoordinates(Np, Nq, [.5*(Np-1.), .5*(Nq-1.)], img_scale)

    # Spherical Geometries.
    sph_img   = g.SphericalCoordinates(s_img_cart)
    sph_n_img = g.SphericalCoordinates(n_img_cart)

    # 0) Propagate the Incident field to the camera plane.
    # FIXME: Implement separately from image_formation.

    # 1) Scattering.
    # Compute the angular spectrum incident on plane 1.
    es_obj = scatter(s_obj_cart, a_p, n_p, nm_obj, lamb, f/M, z)

    # 2) Collection.
    # Compute the electric field strength factor on plane 2.
    es_img = collection(es_obj, s_obj_cart, s_img_cart, nm_obj, NA)

    # 3) Refocus.
    # Compute the electric fields incident on the camera.
    es_cam = refocus(es_img, sph_img, n_disc_grid, p, q, Np, Nq, NA, M, lamb)

    # 4) Image formation.
    # Recombine with plane wave.
    image = image_formation(es_cam, sph_n_img, k_img)

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
    z = 10.
    a_p = 1.0
    n_p = 1.4
    image = debyewolf(z, a_p, n_p, lamb = 0.447, mpp = 0.135, dim = [201,201], NA = 1.45, 
              nm_obj = 1.339, nm_img = 1.0, M = 100, f = 20.*10**5, quiet = True)
    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.gray()
    plt.show()

def test_plots():
    
        # Electric Field Strength After Aperture At P_1.
    plt.imshow(np.hstack([np.abs(es_obj[0]), np.abs(es_obj[1]), np.abs(es_obj[2])]))
    plt.title(r'Electric Field strength  $(r, \theta, \phi)$ at $P_1$ After Aperture')
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()

    # Electric Field Strength at P2.
    plt.imshow(np.hstack([np.abs(es_img[0]), np.abs(es_img[1]), np.abs(es_img[2])]))
    plt.title(r'Electric Field strength $(r, \theta, \phi)$ at $P_2$')
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()

    # Auxiliary Field at P2.
    plt.imshow(np.hstack([np.abs(g_aux[0]), np.abs(g_aux[1]), np.abs(g_aux[2])]))
    plt.title(r'Auxiliary field $(r, \theta, \phi)$ at $P_2$')
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()

    # FFT of Auxiliary field.
    plt.imshow(np.hstack([np.abs(es_m_n[0]), np.abs(es_m_n[1]), np.abs(es_m_n[2])]))
    plt.title(r'Fourier Transform of aux $(r, \theta, \phi)$')
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()

    plt.show()
    '''
    # Electric Field After Dealiasing.
    plt.imshow(np.hstack([np.abs(temp[0]), np.abs(temp[1]), np.abs(temp[2])]))
    plt.title(r'Electric field $(x,y,z)$ at the camera plane after dealiasing')
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()

    plt.show()
    '''
    # Real and Imaginary Components of the image.
    real_part = np.sum(np.real(es_cam_cart), axis = 0)
    imag_part = np.sum(np.imag(es_cam_cart), axis = 0)

    plt.imshow(np.hstack([real_part,imag_part]))
    plt.title('Es_cam_cart real and imaginary')
    plt.show()


if __name__ == '__main__':
    test_debye()
