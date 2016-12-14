import numpy as np

class CartesianCoordinates(object):
    def __init__(self, xy, origin, mask = None, units = None):
        '''
        Encapsulate the method and attributes associated with a coordinate
        system.
        '''
        self.origin = origin
        self.x = xy[1] - origin[0]
        self.y = xy[0] - origin[1]

        self.units = units
        if self.units != None:
            self.x *= units[0]
            self.y *= units[0]

        self.xx, self.yy = np.meshgrid(self.x,self.y)
        self.x = self.xx.ravel()
        self.y = self.yy.ravel()
    
        self.extrema = (self.x.min(), self.y.min(), self.x.max(), self.y.max())

        self.shape = self.xx.shape
        self.ravelshape = self.x.shape

    def rescale(self, units):
        self.x *= units[0]
        self.y *= units[0]
        self.xx *= units[0]
        self.yy *= units[0]

        if self.units != None:
            self.x /= self.units[0]
            self.y /= self.units[0]
        self.units = units
        self.units = units

class SphericalCoordinates(object):
    def __init__(self, x, y, z):
            sintheta = sxx_img**2 + syy_img**2
            costheta = nmp.sqrt(1. - sintheta)
            sintheta = nmp.sqrt(sintheta)



class VectorField(object):
    def __init__(self, coordinates, dim):
        self.coords = coordinates
        self.dim = dim
        self.shape = (dim, coordinates.shape[0], coordinates.shape[1])

    def evaluate_field(self, function):
        '''
        Evaluate field over coordinates.
        '''
        pass


class SphericalVectorField(VectorField):
    def __init__(self, coordinates, dim):
        VectorField.__init__(self, coordinates, dim)
    

def spherical_to_cartesian(es_cam, sintheta, costheta, sinphi, cosphi):
    es_cam_cart = np.zeros(es_cam.shape, dtype = complex)
    es_cam_cart += es_cam
    
    es_cam_cart[0,:,:] =  es_cam[0,:,:] * sintheta * cosphi
    es_cam_cart[0,:,:] += es_cam[1,:,:] * costheta * cosphi
    es_cam_cart[0,:,:] -= es_cam[2,:,:] * sinphi
    
    es_cam_cart[1,:,:] =  es_cam[0,:,:] * sintheta * sinphi
    es_cam_cart[1,:,:] += es_cam[1,:,:] * costheta * sinphi
    es_cam_cart[1,:,:] += es_cam[2,:,:] * cosphi
    
    es_cam_cart[2,:,:] =  es_cam[0,:,:] * costheta - es_cam[1,:,:] * sintheta
    
    return es_cam_cart

if __name__ == '__main__':
    xy = np.ogrid[0:10,0:10]
    origin = [0.0, 0.0]
    cart = CartesianCoordinates(xy, origin, units = [1.0, 'um'])
    
    x = np.tile(np.arange(10, dtype = float), 10)
    y = np.repeat(np.arange(10, dtype = float), 10)
    
    print cart.xx
    print cart.x
    print x
    print cart.y
    print y
    print cart.units
    
    
