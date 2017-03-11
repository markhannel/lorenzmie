import numpy as np
import matplotlib.pyplot as plt
#from lorenzmie import lm_angular_spectrum
import debyewolf as dw
import geometry as g
from spheredhm import spheredhm, spherefield
from sphere_coefficients import sphere_coefficients

class LorenzMieTest(object):
    def __init__(self):
        # Properties common to some of the tests.
        self.sample = [0.5, 1.5, 1.3389] # Sample parameters [a_p, n_p, nm_obj]
        self.lamb = 0.447 # [um]
        self.optical = [1.000, 1.45, 2E3, 100] # Optical Train Properties [nm_img, NA, f, M]
        self.mpp = 0.135 # [um/pix]
        self.dim = [201,201]

    def computeAngSpectrum(self, sx, sy, z, p, q):
        '''Computes the angular spectrum using lm_angular_spectrum.'''
        # Grab necessary variables.
        a_p, n_p, nm_obj = self.sample
        lamb = self.lamb
        nm_img, NA, f, M = self.optical

        # Compute the sphere coefficients for the sample in question.
        ab = sphere_coefficients(a_p, n_p, nm_obj, lamb)
        
        # Only compute the field over the aperture of the entrance pupil.
        inds = np.where(sx**2+sy**2 < (NA/nm_obj*p/2.)**2)[0]

        # Compute the angular Spectrum. Set ang_spec to 0.0 outside of the region inds.
        ang_spec = np.zeros([3, p*q], dtype = complex)
        ang_spec[:, inds] = lm_angular_spectrum(sx[inds], sy[inds], ab, lamb, nm_obj, z)
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
        geom = g.CartesianCoordinates(dim_x, dim_y, [dim_x/2., dim_y/2.], scale = [mpp, mpp])
        sx = geom.xx.ravel()
        sy = geom.yy.ravel()
        self.ang_spec = self.computeAngSpectrum(sx, sy, z_pix, dim_x, dim_y)

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
        
        plt.imshow(np.hstack(map(np.abs, [self.ang_spec[0], self.ang_spec[1], self.ang_spec[2]])))
        plt.show()

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
        es_focal_cart[0, :, :] += 1.0 # Combine with plane wave.
        ang_image = np.sum(np.real(es_focal_cart*np.conj(es_focal_cart)), axis = 0)

        ang_image = ang_image.reshape(dim_y, dim_x)

        # Use spheredhm to generate an image at the focal plane.
        image = spheredhm([0,0,z_pix], a_p, n_p, nm_obj, self.dim, self.mpp, self.lamb)

        # Plot results.
        plt.imshow(np.hstack([ang_image, image]))
        plt.gray()
        plt.show()

    def radialDependence(self):
        '''Tests the radial dependence of lm_ang_spectrum.'''
        # Necessary variables.

        # Compute angular spectrum at three different radial distances.

        # Plot results.

class TestDebyeWolf(object):
    def test_compare_fields(self, z=30.):
        # Necessary parameters.
        a_p = 0.5
        n_p = 1.5
        mpp = 0.135
        NA = 1.45
        lamb = 0.447
        nm_obj = 1.339
        nm_img = 1.0
        M = 100
        field_cam = np.sum(dw.particle_field_camera_plane(z/mpp, a_p, n_p, nm_obj=nm_obj, nm_img=nm_img, NA=NA,
                                    lamb=lamb, mpp=mpp, M=M,
                                    quiet=True), axis=0)
        field_cam_scaled = field_cam * M

        dim = field_cam.shape
        print 'Dim', dim
        field = np.sum(spherefield_holo(dim, [0,0, z/mpp], a_p, n_p, nm_obj, mpp, lamb), axis=0)

        # Visually compare the two.
        fields = [field_cam_scaled, field]
        fieldsAbs = map(np.abs, fields)
        fieldsAng = map(np.angle, fields)

        dw.verbose(np.hstack(fieldsAbs + [fieldsAbs[0] - fieldsAbs[1]]),
                r'Camera Plane field, Focal Plane field and their Difference.',
                gray=True)

        dw.verbose(np.hstack(fieldsAng + [fieldsAng[0] - fieldsAng[1]]),
                   r'Camera Plane phase, Focal Plane phase and their Difference.',
                   gray=True)

    def test_compare_fieldsMatchPhase(self, z=30.):
        # Necessary parameters.
        a_p = 1.5
        n_p = 1.5
        mpp = 0.135
        NA = 1.45
        lamb = 0.447
        nm_obj = 1.339
        nm_img = 1.0
        M = 100
        field_cam = np.sum(dw.particle_field_camera_plane(z/mpp, a_p, n_p, nm_obj=nm_obj, nm_img=nm_img, NA=NA,
                                    lamb=lamb, mpp=mpp, M=M,
                                    quiet=True), axis=0)
        field_cam_scaled = field_cam * M

        dim = field_cam.shape
        print 'Dim', dim
        field = np.sum(spherefield_holo(dim, [0,0, z/mpp], a_p, n_p, nm_obj, mpp, lamb), axis=0)

        phase_cam = np.angle(field_cam_scaled[dim[0]/2, dim[1]/2])
        phase = np.angle(field[dim[0]/2, dim[1]/2])
        print 'phase at center', phase_cam, phase

        field_cam_scaled *= np.exp(-1.j * (phase_cam - phase))

        phase_cam = np.angle(field_cam_scaled[dim[0] / 2, dim[1] / 2])
        phase = np.angle(field[dim[0] / 2, dim[1] / 2])
        print 'phase at center', phase_cam, phase
        # Visually compare the two.
        fields = [field_cam_scaled, field]
        fieldsAbs = map(np.abs, fields)
        fieldsAng = map(np.angle, fields)

        dw.verbose(np.hstack(fieldsAbs + [fieldsAbs[0] - fieldsAbs[1]]),
                r'Camera Plane field, Focal Plane field and their Difference.',
                gray=True)

        dw.verbose(np.hstack(fieldsAng + [fieldsAng[0] - fieldsAng[1]]),
                   r'Camera Plane phase, Focal Plane phase and their Difference.',
                   gray=True)

    def test_image(self, z=30.0, quiet=False):
        from spheredhm import spheredhm

        # Necessary parameters.
        a_p = 0.5
        n_p = 1.5

        NA = 1.45
        lamb = 0.447
        f = 20. * 10 ** 2
        dim = [201, 201]  # FIXME: Does nothing.
        nm_obj = 1.339
        nm_img = 1.0
        M = 100
        mpp = 0.135

        # Produce image with Debye-Wolf Formalism.
        cam_image = dw.image_camera_plane(z / mpp, a_p, n_p, nm_obj=nm_obj,
                                       nm_img=nm_img, NA=NA, lamb=lamb,
                                       mpp=mpp, M=M, f=f, dim=dim,
                                       quiet=quiet)

        # Produce image in the focal plane.
        dim = cam_image.shape
        image = spheredhm([0, 0, z / mpp], a_p, n_p, nm_obj, dim, mpp, lamb)

        print 'max image, max cam_imag', np.max(image), np.max(cam_image)

        # Visually compare the two.
        diff = M ** 2 * cam_image - image
        print("Maximum difference: {}".format(np.max(diff)))
        dw.verbose(np.hstack([M ** 2 * cam_image, image, diff + 1]),
                r'Camera Plane Image, Focal Plane Image and their Difference.',
                gray=True)

    def test_imageStack(self, zRange=(5.0, 40.), nSteps = 10, quiet=True):
        from spheredhm import spheredhm

        # Necessary parameters.
        a_p = 0.5
        n_p = 1.5

        NA = 1.45
        lamb = 0.447
        f = 20. * 10 ** 2
        dim = [201, 201]  # FIXME: Does nothing.
        nm_obj = 1.0
        nm_img = 1.0
        M = 100
        mpp = 0.135

        try:
            images = np.load('imageSlices.npz')
            imageSlice = images['imageSlice']
            imageSliceCam = images['imageSliceCam']
        except:
            # Produce image with Debye-Wolf Formalism.
            cam_image = dw.image_camera_plane(zRange[0] / mpp, a_p, n_p, nm_obj=nm_obj,
                                           nm_img=nm_img, NA=NA, lamb=lamb,
                                           mpp=mpp, M=M, f=f, dim=dim,
                                           quiet=quiet)

            # Produce image in the focal plane.
            dim = cam_image.shape

            imageSliceCam = np.zeros((nSteps, dim[0]))
            imageSlice = np.zeros((nSteps, dim[0]))

            zs = np.linspace(zRange[0], zRange[1], nSteps)
            for i, z in enumerate(zs):
                print 'Calculating slice ', i
                # Produce image with Debye-Wolf Formalism.
                cam_image = dw.image_camera_plane(z / mpp, a_p, n_p, nm_obj=nm_obj,
                                           nm_img=nm_img, NA=NA, lamb=lamb,
                                           mpp=mpp, M=M, f=f, dim=dim,
                                           quiet=quiet)


                image = spheredhm([0, 0, -z / mpp], a_p, n_p, nm_obj, dim, mpp, lamb)

                imageSliceCam[i] = cam_image[:, dim[1]/2]
                imageSlice[i] = image[:, dim[1]/2]


            np.savez('imageSlices.npz', imageSlice=imageSlice, imageSliceCam=imageSliceCam)

        # Save and plot result
        dw.verbose(np.hstack([M ** 2 * imageSliceCam, imageSlice]),
                    r'Image slices versus z.',
                    gray=True, outfile='imageSlices.png',
                    extent=[0, len(imageSliceCam[0,:])*2*mpp, 0, zRange[1]-zRange[0]])


class TestSpheredhm(object):
    def test_belowFocal(self):
        '''Produces a test hologram resulting from a spherical scatterer below focal plane.'''
        # Particle and imaging properties.
        mpp = 0.135
        z = -30.0/mpp
        rp = [0,0,z]
        a_p = 0.5
        n_p = 1.5
        n_m = 1.339
        dim = [201,201]
        lamb = 0.447

        # Produce Image.
        image = spheredhm(rp, a_p, n_p, n_m , dim, lamb = lamb, mpp = mpp)

        # Plot the hologram.
        import matplotlib.pyplot as plt
        plt.imshow(image)
        plt.title("Test Hologram")
        plt.gray()
        plt.show()



def spherefield_holo(dim, rp, a_p, n_p, n_m, mpp, lamb):
    """Convenience function to get scattered field."""
    k = 2 * np.pi * n_m * mpp / lamb
    nx, ny = dim
    x = np.tile(np.arange(nx, dtype=float), ny)
    y = np.repeat(np.arange(ny, dtype=float), nx)
    x -= float(nx) / 2. + float(rp[0])
    y -= float(ny) / 2. + float(rp[1])
    zp = float(rp[2])

    field = spherefield(x, y, zp, a_p, n_p, n_m=n_m, cartesian=True, mpp=mpp,
                        lamb=lamb, precision=False)
    field *= np.exp(np.complex(0., -k * zp))

    return field.reshape(3, int(ny), int(nx))

if __name__ == '__main__':
    # Instantiate test class.
    lt = LorenzMieTest()

    # Run tests.
    lt.vsSpheredhm()

    
