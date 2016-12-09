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
        if self.units != None:
            self.x /= units[0]
            self.y /= units[0]
        self.units = units
        self.units = units

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


if __name__ == '__main__':
    xy = np.ogrid[0:10,0:10]
    origin = [0.0, 0.0]
    cart = CartesianCoordinates(xy, origin, units = [2.0, 'um'])
    
    print cart.xx
    print cart.x
    print cart.units
    
    
