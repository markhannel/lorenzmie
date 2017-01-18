import numpy as np
import matplotlib.pyplot as plt
from lorenzmie import lm_angular_spectrum
import geometry as g
from spheredhm import spheredhm
from sphere_coefficients import sphere_coefficients

class LorenzMieTest(object):
    def __init__(self):
        # Properties common to some of the tests.
        self.sample = [0.5, 1.5, 1.3389] # Sample parameters [a_p, n_p, nm_obj]
        self.lamb = 0.447 # [um]
        self.optical = [1.000, 1.45, 2E5, 100] # Optical Train Properties [nm_img, NA, f, M]
        self.mpp = 0.135 # [um/pix]
        self.dim = [201,201]

    def computeAngSpectrum(self, sx, sy, z, p, q):
        '''Computes the angular spectrum using lm_angular_spectrum.'''
        # Grab necessary variables.
        a_p, n_p, nm_obj = self.sample
        lamb = self.lamb
        nm_img, NA, f, M = self.optical
        r = f/M

        # Compute the sphere coefficients for the sample in question.
        ab = sphere_coefficients(a_p, n_p, nm_obj, lamb)
        
        # Only compute the field over the aperture of the entrance pupil.
        inds = np.where(sx**2+sy**2 < (NA/nm_obj*p/2.)**2)[0]

        # Compute the angular Spectrum. Set ang_spec to 0.0 outside of the region inds.
        ang_spec = np.zeros([3, p*q], dtype = complex)
        ang_spec[:, inds] = lm_angular_spectrum(sx[inds], sy[inds], ab, lamb, nm_obj, f/M, z)
        return ang_spec.reshape(3, p, q)
        
    def vsSpheredhm(self):
        '''Compares the image computed with lm_ang_spectrum to the image computed with 
        spheredhm.'''

        # Necessary variables.
        a_p, n_p, nm_obj = self.sample
        z = 12.0 # [um]
        k = 2.0*np.pi/(self.lamb/np.real(nm_obj)/self.mpp) # wavenumber in medium [pixel^-1]
        mpp = self.mpp
        dim_x, dim_y = self.dim
        z_pix = z/self.mpp

        # Compute the angular spectrum with the z offset set to 0.
        z_offset = 0. # [um]        
        geom = g.CartesianCoordinates(dim_x, dim_y, [-dim_x/2., -dim_y/2.], scale = [mpp, mpp])
        sx = geom.xx.ravel()
        sy = geom.yy.ravel()
        self.ang_spec = self.computeAngSpectrum(sx, sy, z_offset, dim_x, dim_y)

        # Use the angular spectrum to find the field at the focal plane.
        rho = np.sqrt(geom.xx**2 + geom.yy**2)
        r = np.arctan2(z_pix, rho)
        phi = np.arctan2(geom.yy, geom.xx)
        theta = np.arctan2(rho, z_pix)
        sintheta = np.sin(theta)
        costheta = np.cos(theta)
        cosphi = np.cos(phi)
        sinphi = np.sin(phi)

        es_focal = self.ang_spec*np.exp(-1j*k*r)/r

        # Convert es_focal to spherical coordinates
        es_focal_cart = np.zeros(es_focal.shape, dtype = complex)
        es_focal_cart += es_focal
    
        es_focal_cart[0,:,:] =  es_focal[0,:,:] * sintheta * cosphi
        es_focal_cart[0,:,:] += es_focal[1,:,:] * costheta * cosphi
        es_focal_cart[0,:,:] -= es_focal[2,:,:] * sinphi
    
        es_focal_cart[1,:,:] =  es_focal[0,:,:] * sintheta * sinphi
        es_focal_cart[1,:,:] += es_focal[1,:,:] * costheta * sinphi
        es_focal_cart[1,:,:] += es_focal[2,:,:] * cosphi
    
        es_focal_cart[2,:,:] =  es_focal[0,:,:] * costheta - es_focal[1,:,:] * sintheta
        
        # Combine the scattered field and the incident field to produce an image.
        es_focal_cart *= np.exp(np.complex(0.,-k*z_pix)) # Plane wave's phase difference.
        es_focal_cart[:,0] += 1.0 # Combine with plane wave.
        ang_image = np.sum(np.real(es_focal_cart*np.conj(es_focal_cart)), axis = 0)

        ang_image = ang_image.reshape(dim_y, dim_x)
        print type(dim_y)
        # Use spheredhm to generate an image at the focal plane.
        image = spheredhm([0,0,z_pix], a_p, n_p, nm_obj, self.dim, self.mpp, self.lamb)

        # Plot results.
        plt.imshow(np.hstack([ang_image, image]))
        plt.gray()
        plt.show()

if __name__ == '__main__':
    # Instantiate test class.
    lt = LorenzMieTest()

    # Run tests.
    lt.vsSpheredhm()

    
