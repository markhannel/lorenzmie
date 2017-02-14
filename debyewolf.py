import numpy as np
from sphericalfield import sphericalfield
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

def displacement(sxx, syy, z, k):
    '''Returns the displacement phase accumulated by an angular spectrum
    propagating a distance z. 
    
    Ref[2]: J. Goodman, Introduction to Fourier Optics, 2nd Edition, 1996
            [See 3.10.2 Propagation of the Angular Spectrum]
    '''
    return np.exp(1.0j * k * z * np.sqrt( 1. - sxx**2 - syy**2))

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
    

def consv_energy(es, s_obj, s_img, r_max, M):
    '''
    Changes electric field strength factor density to obey the conversation of 
    energy. See Eq. 108 of Ref. 1.
    '''

    cos_theta_obj = s_obj.costheta
    cos_theta_img = s_img.costheta

    # Obey the conservation of energy and make use of the abbe-sine condition. Eq. 108. Ref. 1.
    # FIXME (MDH): You did not magnify by M. Is it really the case that
    # the incident field and the scattered field are similarly magnified
    es[:,:,:] *= -np.sqrt(M*cos_theta_img/cos_theta_obj)

    return es

def remove_r(es):
    '''Remove r component of vector.'''
    es[0,:,:] = 0.0
    return es

def propagate_plane_wave(amplitude, k, path_len, shape):
    '''Propagates a plane with wavenumber k through a distance path_len. 
    The wave is polarized in the x direction. The field is given as a 
    cartesian vector field.'''
    e_inc = np.zeros(shape, dtype = complex)
    e_inc[0,:,:] += amplitude*np.exp(1.j * k * path_len)
    return e_inc

def scatter(s_obj_cart, a_p, n_p, nm_obj, lamb, r, mpp):
    '''Compute the angular spectrum arriving at the entrance pupil.'''
    
    p, q = s_obj_cart.shape

    lamb_m = lamb/np.real(nm_obj)/mpp
    ab = sphere_coefficients(a_p, n_p, nm_obj, lamb)
    sx = s_obj_cart.xx.ravel()
    sy = s_obj_cart.yy.ravel()

    # Compute the electromagnetic strength factor on the object side 
    # (Eq 40 Ref[1]).
    ang_spec = sphericalfield(sx*r, sy*r, r, ab, lamb_m, cartesian=False, 
                              str_factor=True)

    return ang_spec.reshape(3, p, q)

def collection(ang_spec, s_obj_cart, s_img_cart, nm_obj, NA, M):
    '''Compute the angular spectrum leaving the exit pupil.'''

    # Ensure conservation of energy is observed with abbe sine condition.
    es_img = consv_energy(ang_spec, s_obj_cart, s_img_cart, NA/nm_obj, M)
    es_img = remove_r(es_img) # Should be no r component.
    es_img = np.nan_to_num(es_img)

    return es_img

def refocus(es_img, s_img, n_disc_grid, p, q, Np, Nq, NA, M, lamb):
    '''Propagates the electric field from the exit pupil to the image plane.'''
        
    # Compute auxiliary (Eq. 133) with zero padding!
    # FIXME (FUTURE): aber  = np.zeros([3, Np, Nq], complex) # As a function of sx_img, sy_img
    g_aux = np.zeros([3, p, q], complex)
    for i in xrange(3):
        g_aux[i, :,:] = es_img[i,:,:]/s_img.costheta
        #g_aux *= np.exp(-1.j*k_img*aber)

    # Apply discrete Fourier Transform (Eq. 135).
    es_m_n = np.fft.fft2(g_aux, s = (Np,Nq))

    for i in xrange(3):
        es_m_n[i] = np.fft.fftshift(es_m_n[i])

    # Compute the electric field at plane 3.
    # Accounting for aliasing.
    es_cam  = (1.j*NA**2/(M*lamb))*(4./(p*q))*es_m_n 
    # FIXME (MDH): Should it be p*q or Np*Nq

    mm = n_disc_grid.xx
    nn = n_disc_grid.yy

    for i in xrange(3):
        es_cam[i,:,:] *= np.exp(-1.j*np.pi*( mm*(1.-p)/Np + nn*(1.-q)/Nq))

    return es_cam

def image_formation(es_cam, e_inc_cam):
    '''Produces an image from the electric fields present.'''
    fields = es_cam + e_inc_cam
    image = np.sum(np.real(fields*np.conjugate(fields)), axis = 0)
    return image

def debyewolf(z, a_p, n_p,  nm_obj=1.339, nm_img=1.0, NA=1.45, lamb=0.447, 
              mpp=0.135, M=100, f=2.E5, dim=[201,201], quiet=True):
    '''
    Returns an image in the camera plane due to a spherical scatterer with 
    radius a_p and refractive index n_p at a height z above the focal plane. 

    Args:
        z:     [um] scatterer's distance from the focal plane.
        a_p:   [um] sets the radius of the spherical scatterer.
        n_p:   [R.I.U.] sets the refractive index of the scatterer.
        nm_obj:[R.I.U.] sets the refractive index of medium immersing the 
               scattered.
               Default: 1.339 (Water)
        nm_img:[R.I.U.] sets the refractive index of the medium immersing the 
               camera.
               Default: 1.00 (Air)
        NA:    [unitless] The numerical aperture of the optical train.
               Default: 1.45 (100x Nikon Lambda Series)
        lamb:  [um] wavelength of the incident illumination.
               Default: 0.447 (Coherent Cube.. blue)
        mpp:   [um/pix] sets the size of a pixel.
               Default: 0.135.
        M:     [unitless] Magnification of the optical train.
`              Default: 100
        f:     [um]: focal length of the objective. Sets the distance between 
               the entrance
               pupil and the focal plane.
               Default: 20E5 um.
        dim:   [nx, ny]: (will) set the size of the resulting image.
               Default: [201,201]

    Return:
        image: [?, ?] - Currently dim is not implemented. The resulting image 
               size is dictated by the padding chose for the fourier transform.

    Ref[1]: Capoglu et al. (2012). "The Microscope in a Computer:...", 
               Applied Optics, 38(34), 7085.
    '''

    # Necessary constants.
    k_img = 2*np.pi*nm_img/lamb*mpp # [pix**-1]
    k_obj = 2*np.pi*nm_obj/lamb*mpp # [pix**-1]
    r_max = 100. # [pix] 

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
    n_img_cart  = g.CartesianCoordinates(Np, Nq, [.5*(Np-1.), .5*(Nq-1.)], 
                                         img_scale)

    # Spherical Geometries.
    s_obj_cart.acquire_spherical(1.)
    s_img_cart.acquire_spherical(1.)
    n_img_cart.acquire_spherical(1.)

    # 0) Propagate the Incident field to the camera plane.
    e_inc = propagate_plane_wave(1.0, k_img, 0, (3, Np, Nq))

    # 1) Scattering.
    # Compute the angular spectrum incident on entrance pupil of the objective.
    ang_spec = scatter(s_obj_cart, a_p, n_p, nm_obj, lamb, r_max, mpp)

    if quiet == False:
        plt.imshow(np.hstack(map( np.abs, ang_spec[:])))
        plt.title(r'After Scatter $(r,\theta,\phi)$')
        plt.show()

    # 1.5) Displacing the field.
    # Propagate the angular spectrum a distance z_p.
    sxx = s_obj_cart.xx
    syy = s_obj_cart.yy

    inside = sxx**2+syy**2 <= 1.
    disp = displacement(sxx, syy, z, k_obj)
    disp[~inside] = 1
    for i in xrange(1,3):
        ang_spec[i, inside] *= disp[inside]

    plt.imshow(map( np.real, disp))
    plt.title(r'Displacement field')
    plt.show()

    # 2) Collection.
    # Compute the electric field strength factor leaving the tube lens.
    es_img = collection(ang_spec, s_obj_cart, s_img_cart, nm_obj, NA, M)

    if quiet == False:
        plt.imshow(np.hstack(map( np.abs, es_img[:])))
        plt.title(r'After Collection ($r$, $\theta$, $\phi$)')
        plt.show()

    '''
    # 2.5) Displacement.
    # Propagate the angular spectrum a distance z_p.
    # FIXME (MDH): Should displacement take place before 
    sxx = s_img_cart.xx
    syy = s_img_cart.yy
    inds = np.where(sxx**2+syy**2 <= 1.)
    disp = np.ones([3, p, q], dtype = complex)
    for i in xrange(1,3):
        disp[i, inds[0], inds[1]] = displacement(sxx[inds[0], inds[1]], syy[inds[0], inds[1]], z, lamb)
    es_img[:, inds[0], inds[1]] *= disp[:, inds[0], inds[1]]
    '''
    # 3) Refocus.
    # Input the electric field strength into the debye-wolf formalism to 
    # compute the scattered field at the camera plane.
    es_img = g.spherical_to_cartesian(es_img, s_img_cart)
    es_cam = refocus(es_img, s_img_cart, n_disc_grid, p, q, Np, Nq, NA, M, lamb)

    if quiet == False:
        plt.imshow(np.hstack(map( np.abs, es_cam[:])))
        plt.title(r'After Refocusing $(x, y, z)$')
        plt.show()

    # 4) Image formation.
    # Combine the electric fields in the image plane to form an image.
    image = image_formation(es_cam, e_inc)

    return image

def test_discretize():
    NA = 1.45
    M = 100
    lamb = 0.447
    nm_img = 1.0
    mpp = 0.135
    pad_p, pad_q, p, q = discretize_plan(NA, M, lamb, nm_img, mpp)
    
    del_x = lamb*p*M/(2*NA*(pad_p+p))
    print del_x/M

def test_debye():
    import matplotlib.pyplot as plt
    from spheredhm import spheredhm

    # Necessary parameters.
    z = 10.
    a_p = 0.5
    n_p = 1.5
    mpp = 0.135

    # Produce image with Debye-Wolf Formalism.
    deb_image = debyewolf(z/mpp, a_p, n_p,  nm_obj = 1.339, nm_img = 1.0,  
                          NA = 1.45, lamb = 0.447, mpp = 0.135, M = 100, 
                          f = 20.*10**2, dim = [201,201], quiet = False)
    plt.imshow(deb_image)
    plt.title(r'Hologram with Debye-Wolf')
    plt.gray()
    plt.show()
    
    # Produce image in the focal plane.
    dim = deb_image.shape
    image = spheredhm([0,0, z/mpp], a_p, n_p, 1.339, dim, 0.135, 0.447)

    # Visually compare the two.
    plt.imshow(np.hstack([deb_image, image]))
    plt.title(r'Comparing Debye-Wolf Hologram to Spheredhm Hologram')
    plt.gray()
    plt.show()

if __name__ == '__main__':
    test_debye()
