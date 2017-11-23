import debyewolf as dw
import numpy as np
import numpy.testing as npt
import geometry as g
import matplotlib.pyplot as plt
import spheredhm as sph
import azimedian as azi

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
    mpp_r, Np, Nq, p, q = dw.discretize_plan(NA, M, lamb, nm_img, mpp)
    
    del_x = lamb*p/(2*NA*(Np))
    print del_x
    assert np.round(del_x, decimals=4) == mpp

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
    z = 100. # [pix]
    a_p = 0.5 # [um]
    n_p = 1.59 # [arb] 
    nm = 1.339 # [arb]
    nm_obj = 1.339 # [arb]
    nm_img = 1.339 # [arb]
    lamb = 0.447 # [um]
    mpp = 0.1350981 # [um/pix]
    NA = 1.45
    M = 1. # [arb]
    
    # Image formation.
    cam_image = dw.image_camera_plane(z, a_p, n_p, nm, nm_obj=nm_obj, nm_img=nm_img,
                                      NA=NA, lamb=lamb, mpp=mpp, M=M, quiet=True)
    cam_image *= M**2*nm_img/nm_obj

    # Produce image in focal plane with spheredhm.
    dim = cam_image.shape
    focal_image = sph.spheredhm([-0.5, -0.5, z], a_p, n_p, nm_obj, dim, mpp, lamb)
    
    # PYTEST
    npt.assert_array_almost_equal(focal_image, cam_image, decimal = 2)
    
    '''
    # Temporary debugging information.
    # Numerical information.
    print('Mean and max value of camera image:      {} {}'.format(np.mean(cam_image), np.max(cam_image)))

    print('Mean and max value of focal plane image: {} {}'.format(np.mean(focal_image), np.max(focal_image)))   
    print('Max value of normalized camera image:      {}'.format(np.max(cam_image)/np.mean(cam_image)))

    print('Max value of normalized focal plane image: {}'.format(np.max(focal_image)/np.mean(focal_image)))   
    

    # 2D images
    dw.verbose(np.hstack([cam_image, focal_image, 1 + cam_image - focal_image]), 'Camera Image, Focal Image, Difference', gray = True)

    
    xc, yc = (dim[0]-1.)/2, (dim[1]-1)/2.
    cam_rad = azi.azimedian(cam_image)
    focal_rad = azi.azimedian(focal_image) 
    # FIXME (MDH): center is not correct...

    end = 150
    plt.plot(cam_rad[:end], 'r', label = 'Camera Plane')
    plt.plot(focal_rad[:end], 'black', label = 'Focal Plane')
    plt.xlabel('Radial distance [pix]')
    plt.ylabel('Normalized Intensity [arb]')
    plt.legend()
    plt.show()
    '''

if __name__ == '__main__':
    test_propagate_ang_spec_microscope()
