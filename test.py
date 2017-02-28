import numpy as np
import matplotlib.pyplot as plt
import geometry as g
from debyewolf import image_formation, collection, refocus

def fict_ang_spec(obj_geom, img_geom, M, p, q):
    '''Returns convenient fictitious angular spectrum for testing purposes.'''
    # Geometrical factors.
    costheta_obj = obj_geom.costheta
    costheta_img = img_geom.costheta

    # Produce angular spectrum.
    ang_spec = np.zeros([3, p, q], dtype = complex)
    ang_spec[1:, :] += 1./M*np.sqrt(costheta_obj*costheta_img) # r-component is zero.

    return ang_spec

def image_focal_plane(ang_spec, geom, z, k):
    '''Produces an image from the fictitious scatterer in the focal plane.'''
    # Propagate scattered field to the focal plane (z away).
    r = None
    e_field = ang_spec*np.exp(1.j*k*r)/r
    #e_field = g.spherical_to_cartesian(e_field, geom)

    # Propagate incident field to the focal plane (z away).
    e_inc = 1.0*np.exp(1.j*k*z)

    return image_formation(e_inc, e_image)

def image_camera_plane(ang_spec, s_obj_cart, s_img_cart, n_disc_grid, p, q, Np, Nq, 
                       nm_obj, NA, M, lamb):
    '''Produces an image of the fictitious scatter in the camera plane.'''

    ### Propagate the scattered field to the camera plane.
    # Conservation of energy between ent and exit pupil.
    es_img = collection(ang_spec, s_obj_cart, s_img_cart, nm_obj, M)
    #es_img = g.spherical_to_cartesian(es_img, s_img_cart)
    
    # Propagate from the exit pupil to the camera plane (Debye-Wolf).
    es_cam = refocus(es_img, s_img_cart, n_disc_grid, p, q, Np, Nq, NA, M, lamb)
    
    return es_cam

def test_fic_ang_spec():
    # Parameters
    nm_obj = 1.339
    nm_img = 1.0 
    NA = 1.45 
    lamb = 0.447 
    mpp = 0.135 
    M = 100 
    f = 2.E5

    # Arbitrary size of image
    p, q = 20, 20
    Np, Nq = 250, 250

    # Compute the two geometries, s_img, s_obj, n_img
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

    # Spherical geometries.
    s_img_cart.acquire_spherical(1.)
    s_obj_cart.acquire_spherical(1.)
    
    # Compute angular spectrum.
    ang_spec = fict_ang_spec(s_obj_cart, s_img_cart, M, p, q)

    # View Angular Spectrum
    plt.imshow(np.hstack(map(np.abs, ang_spec[:])))
    plt.show()

    # Propagate the field to the camera plane.
    image = image_camera_plane(ang_spec, s_obj_cart, s_img_cart, n_disc_grid, p, q, Np, Nq, 
                       nm_obj, NA, M, lamb)

    plt.imshow(np.hstack(map(np.real, image)))
    plt.show()

if __name__ == '__main__':
      test_fic_ang_spec()
