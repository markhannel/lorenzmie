import numpy as np
from lorenzmie import lm_angular_spectrum
from sphere_coefficients import sphere_coefficients
import matplotlib.pyplot as plt
import geometry as g

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
    Changes electric field strength factor density to obey the conversation of energy. 
    See Eq. 108 of Ref. 1.

    FIXME (MDH): Should spherical coordinates be done differently?
    '''

    # Determine the
    sx_obj = s_obj.xx
    sy_obj = s_obj.yy
    sx_img = s_img.xx
    sy_img = s_img.yy
    r_2 = sx_obj**2 +sy_obj**2
    inds = np.where(r_2 <= r_max**2)
    cos_theta_obj = np.zeros(sx_obj.shape)

    # Compute the necessary cosine terms in Eq 108.
    cos_theta_img = np.sqrt(1. - (sx_img**2+sy_img**2))
    cos_theta_obj[inds] = np.sqrt(1. - (sx_obj[inds]**2+sy_obj[inds]**2))

    # Obey the conservation of energy and make use of the abbe-sine condition. Eq. 108. Ref. 1.
    es[:,inds[0],inds[1]] *= np.sqrt(cos_theta_img[inds[0],inds[1]]/cos_theta_obj[inds[0],inds[1]])

    return es

def remove_r(es):
    '''Remove r component of vector.'''
    es[0,:,:] = 0.0
    return es
  
    
def scatter(s_obj_cart, a_p, n_p, nm_obj, NA, lamb, r, z):
    '''Compute the angular spectrum arriving at the entrance pupil.'''
    
    
    ab = sphere_coefficients(a_p, n_p, nm_obj, lamb)
    sx = s_obj_cart.xx.ravel()
    sy = s_obj_cart.yy.ravel()
    
    sx *= r
    sy *= r
    p, q = s_obj_cart.shape

    # Compute the electromagnetic strength factor on the object side (Eq 40 Ref[1]). 
    # By default, ang_spec = 0 on all points where sx**2 + sy**2 >= (NA/nm_obj)**2
    inds = np.where(sx**2+sy**2 < (NA/nm_obj)**2)[0]
    ang_spec = np.zeros([3, p*q], dtype = complex)
    ang_spec[:,inds] = lm_angular_spectrum(sx[inds], sy[inds], ab, lamb, nm_obj, r, z)

    return ang_spec.reshape(3, p, q)

def collection(ang_spec, s_obj_cart, s_img_cart, nm_obj, NA):
    '''Compute the angular spectrum leaving the exit pupil.'''

    # Ensure conservation of energy is observed with abbe sine condition.
    es_img = consv_energy(ang_spec, s_obj_cart, s_img_cart, NA/nm_obj)
    es_img = remove_r(es_img) # Should be no r component.
    es_img = np.nan_to_num(es_img)

    return es_img

def refocus(es_img, sph_img, n_disc_grid, p, q, Np, Nq, NA, M, lamb):
    '''Propagates the electric field from the exit pupil to the image plane.'''
        
    # Compute auxiliary (Eq. 133) with zero padding!
    # FIXME (FUTURE): aber  = np.zeros([3, Np, Nq], complex) # As a function of sx_img, sy_img
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

        
    # Auxiliary Field at P2.
    plt.imshow(np.hstack([np.abs(es_cam[0]), np.abs(es_cam[1]), np.abs(es_cam[2])]))
    plt.title(r'es_cam $(r, \theta, \phi)$ at $P_2$') 
    plt.show()



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

def debyewolf(z, a_p, n_p,  nm_obj = 1.339, nm_img = 1.0,  NA = 1.45, lamb = 0.447, mpp = 0.135, 
              M = 100, f = 2.E5, dim = [201,201]):
    '''
    Returns an image in the camera plane due to a spherical scatterer with radius a_p and 
    refractive index n_p at a height z above the focal plane. 

    Args:
        z:     [um] scatterer's distance from the focal plane.
        a_p:   [um] sets the radius of the spherical scatterer.
        n_p:   [R.I.U.] sets the refractive index of the scatterer.
        nm_obj:[R.I.U.] sets the refractive index of medium immersing the scattered.
               Default: 1.339 (Water)
        nm_img:[R.I.U.] sets the refractive index of the medium immersing the camera.
               Default: 1.00 (Air)
        NA:    [unitless] The numerical aperture of the optical train.
               Default: 1.45 (100x Nikon Lambda Series)
        lamb:  [um] wavelength of the incident illumination.
               Default: 0.447 (Coherent Cube.. blue)
        mpp:   [um/pix] sets the size of a pixel.
               Default: 0.135.
        M:     [unitless] Magnification of the optical train.
               Default: 100
        f:     [um]: focal length of the objective. Sets the distance between the entrance
               pupil and the focal plane.
               Default: 20E5 um.
        dim:   [nx, ny]: (will) set the size of the resulting image.
               Default: [201,201]

    Return:
        image: [?, ?] - Currently dim is not implemented. The resulting image size is dictated
               by the padding chose for the fourier transform.

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
    
    # Scale factors for the coordinate systems.
    img_factor = 2*NA/(M*nm_img)
    obj_factor = 2*NA/nm_obj
    img_scale = [img_factor*1./p, img_factor*1./q]
    obj_scale = [obj_factor*1./p, obj_factor*1./q]

    # Cartesian Geometries.
    # FIXME (MDH): is it necessary to have s_obj_cart?
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
    # Compute the angular spectrum incident on entrance pupil of the objective.
    ang_spec = scatter(s_obj_cart, a_p, n_p, nm_obj, NA, lamb, f/M, z)

    # Auxiliary Field at P2.
    plt.imshow(np.hstack([np.real(ang_spec[0]), np.real(ang_spec[1]), np.real(ang_spec[2])]))
    plt.title(r'ang_spec before $(r, \theta, \phi)$ at $P_2$') 
    plt.show()

    # 2) Collection.
    # Compute the electric field strength factor leaving the tube lens.
    es_img = collection(ang_spec, s_obj_cart, s_img_cart, nm_obj, NA)

    # 3) Refocus.
    # Input the electric field strength into the debye-wolf formalism to compute the 
    # scattered field at the camera plane.
    es_cam = refocus(es_img, sph_img, n_disc_grid, p, q, Np, Nq, NA, M, lamb)

    # 4) Image formation.
    # Combine the electric fields in the image plane to form an image.
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
    import matplotlib.pyplot as plt
    from spheredhm import spheredhm

    # Necessary parameters.
    z = 10.
    a_p = 1.0
    n_p = 1.4

    # Produce image with Debye-Wolf Formalism.
    deb_image = debyewolf(z, a_p, n_p,  nm_obj = 1.339, nm_img = 1.0,  NA = 1.45, 
                      lamb = 0.447, mpp = 0.135, M = 100, f = 20.*10**2, dim = [201,201])
    plt.imshow(deb_image)
    plt.title('Hologram with Debye-Wolf')
    plt.gray()
    plt.show()
    
    # Produce image in the focal plane.
    dim = deb_image.shape
    image = spheredhm([0,0,z/0.135], a_p, n_p, 1.339, dim, 0.135, 0.447)

    # Visually compare the two.
    plt.imshow(np.hstack([deb_image, image]))
    plt.title('Comparing Debye-Wolf Hologram to Spheredhm Hologram')
    plt.gray()
    plt.show()

if __name__ == '__main__':
    test_debye()
