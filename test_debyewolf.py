import debyewolf as dw
import numpy as np
import numpy.testing as npt
import geometry as g
import matplotlib.pyplot as plt
import spheredhm as sph

def test_propagate_plane_wave():
    shape = (3,1,1)

    # Propagated through zero distance.
    e_field = dw.propagate_plane_wave(1.0, 1.0, 0.0, shape)
    e_inc = np.zeros(shape, dtype=complex)
    e_inc[0,:,:] += 1.0
    npt.assert_array_equal(e_field, e_inc)
    
    # Propagated through half a phase.
    e_field = dw.propagate_plane_wave(1.0, np.pi/2, 1.0, shape)
    e_inc = np.zeros(shape, dtype=complex)
    e_inc[0,:,:] += 1.j
    npt.assert_array_almost_equal(e_field, e_inc) # almost == up to machine error.
    
def test_discretize_plan():
    # Test that discretize plan produces pixel size which is approximately 
    # equal to mpp*M
    NA = 1.45
    M = 100
    lamb = 0.447
    nm_img = 1.0
    mpp = 0.135
    pad_p, pad_q, p, q = dw.discretize_plan(NA, M, lamb, nm_img, mpp)
    
    del_x = lamb*p*M/(2*NA*(pad_p+p))
    assert np.round(del_x, decimals=1) == mpp*M

def test_collection():
    shape = (3, 100, 100)
    es = np.ones(shape)
    p, q = shape[1], shape[2]
    origin = [.5*(p-1.), .5*(q-1.)]
    
    # if geometries are the same => no theta dependence.
    M = 100. # In reality, if geoms are the same, M = 1.
    nm_obj = 1.4
    nm_img = 1.0
    scale = [2./p, 2./q]
    s_obj = g.CartesianCoordinates(p, q, origin, scale)
    s_obj.acquire_spherical(1.)
    es = dw.collection(es, s_obj, s_obj, nm_obj, nm_img, M)
    result = np.ones(shape)*-M*np.sqrt(nm_img/nm_obj)
    npt.assert_array_almost_equal(es, result)

    # if M = 1 and nm_img == nm_obj, field is just flipped.
    M = 1
    nm_obj = 1.0
    nm_img = 1.0
    es = np.ones(shape)  # Reset field.
    es = dw.collection(es, s_obj, s_obj, nm_obj, nm_img, M)
    npt.assert_array_equal(es, -1*np.ones(es.shape))

def test_propagate_ang_spec_microscope():
    '''
    Test that refocus produces the same image in the camera plane 
    '''
    # Necessary physical parameters.
    z = 200. # [pix]
    a_p = 0.5 # [um]
    n_p = 1.5 # [arb] 
    nm_obj = 1.339 # [arb]
    nm_img = 1.339 # [arb]
    lamb = 0.447 # [um]
    mpp = 0.135 # [um/pix]
    NA = 1.45
    M = 1. # [arb]

    k_obj = 2 * np.pi * nm_obj / lamb * mpp  # [pix**-1]
    r_max = 1000.  # [pix]
    sintheta_img = NA / (M * nm_img)

    # Devise a discretization plan.
    pad_p, pad_q, p, q = dw.discretize_plan(NA, M, lamb, nm_img, mpp)
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
    s_obj_cart = g.CartesianCoordinates(p, q, origin, obj_scale)
    s_img_cart = g.CartesianCoordinates(p, q, origin, img_scale)
    n_disc_grid = g.CartesianCoordinates(Np, Nq, origin=[.5*(Np-1), .5*(Nq-1.)])
    n_img_cart  = g.CartesianCoordinates(Np, Nq, [.5*(Np-1.), .5*(Nq-1.)], 
                                         img_scale)

    # Spherical Geometries.
    s_obj_cart.acquire_spherical(1.)
    s_img_cart.acquire_spherical(1.)
    n_img_cart.acquire_spherical(1.)

    # Compute the angular spectrum incident on entrance pupil of the objective.
    ang_spec = dw.scatter(s_obj_cart, a_p, n_p, nm_obj, lamb, r_max, mpp)

    # Propagate the angular spectrum a distance z_p.
    disp = dw.displacement(s_obj_cart, z, k_obj)
    ang_spec[1:, :] *= disp

    # Collection and refocus.
    es_cam = dw.propagate_ang_spec_microscope(ang_spec, s_obj_cart, s_obj_cart, 
                                              nm_obj, nm_img, M, n_disc_grid,
                                              p, q, Np, Nq, NA, lamb, mpp, 
                                              quiet=True)

    # Image formation.
    e_inc = dw.incident_field_camera_plane(nm_obj, nm_img, lamb, mpp, NA, M, z)

    cam_image = dw.image_formation(es_cam, e_inc)

    # Produce image in focal plane with spheredhm.
    dim = cam_image.shape
    focal_image = sph.spheredhm([0, 0, z], a_p, n_p, nm_obj, dim, mpp, lamb)
    
    print np.max(focal_image)
    print np.max(cam_image)

    del_x = lamb*p*M/(2*NA*(pad_p+p))
    print del_x

    plt.imshow(cam_image-focal_image)
    plt.gray()
    plt.show()

if __name__ == '__main__':
    test_propagate_ang_spec_microscope()
