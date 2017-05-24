from lmfit import Minimizer, Parameters, report_fit
import numpy as np
import spheredhm as sph
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

def mie_loss(params, image, dim):
    """Returns the residual between the image and our Mie model."""
    p = params.valuesdict()
    mie_model = sph.spheredhm([p['x'], p['y'], p['z']], p['a_p'],
                              p['n_p'], p['n_m'], dim,
                              mpp=p['mpp'], lamb=p['lamb'])
    return mie_model - image

class Mie_Fitter(object):
    '''
    Mie_Fitter provides a method for fitting holographic images to the
    Lorenz-Mie theory of light scattering via the Levenberg-Marquardt
    algorithm.

    Inputs:
        init_params: a dictionary containing the initial values for
            each parameter.

    Attributes:
        p: lmfit parameters relevant to the scattering calculation.
            p['x']: x position [pix]
            p['y']: y position [pix]
            p['z']: z position [pix]
            p['a_p']: radius [um]
            p['n_p']: refractive index of scatterer [unitless]
            p['n_m']: refractive index of medium [unitless]
                (Default: 1.3371 water w/ lambda ~ 447 nm)
            p['mpp']: camera pixel calibration [um/pix]
            p['lamb']: wavelength [um]

        result: lmfit result object (or None before fitting procedure).
            result contains the parameter estimates, standard devs,
            covariances, . (See lmfit result object docs).
    '''
    def __init__(self, init_params, fixed=['n_m', 'mpp', 'lamb']):
        # Instantiate parameters.
        self.__init_params__()

        # Set initial values.
        for name, value in init_params.items():
            self.set_param(name, value)

        # Set parameters which should NOT be varied.
        for name in fixed:
            self.fix_param(name)
        
        self.result = None

    def __init_params__(self):
        self.p = Parameters()
        params = ['x', 'y', 'z', 'a_p', 'n_p', 'n_m', 'mpp', 'lamb']
        for p in params:
            self.p.add(p)

    def set_param(self, name, value):
        """Set parameter 'name' to 'value'"""
        self.p[name].value = value

    def fix_param(self, name, choice=False):
        """Fix parameter 'name' to not vary during fitting"""
        self.p[name].vary = choice

    def fit(self, image):
        """Fit a image of a hologram with the current attribute 
        parameters.

        Example:
        >>> p = {'x':0, 'y':0, 'z':100, 'a_p':0.5, 'n_p':1.5, 'n_m':1.337, 
        ...      'mpp':0.135, 'lamb':0.447}
        >>> mie_fit = Mie_Fitter(p)
        >>> mit_fit.result(image)
        """
        dim = image.shape
        minner = Minimizer(mie_loss, self.p, fcn_args=(image, dim))
        self.result = minner.minimize()
        return self.result

        
def example():
    # create data to be fitted
    x,y,z = 0., 0., 100.
    a_p = 0.5
    n_p = 1.5
    n_m = 1.339
    dim = [201,201]
    lamb = 0.447
    mpp = 0.135
    image = sph.spheredhm([x,y,z], a_p, n_p, n_m, dim, mpp, lamb)
    
    # Add noise.
    std = 0.03
    image += np.random.normal(size=image.shape)*std

    init_params = {'x':x, 'y':y, 'z':z, 'a_p':a_p, 'n_p':n_p, 'n_m':n_m,
                   'mpp':mpp, 'lamb':lamb}
    mie_fit = Mie_Fitter(init_params)
    result = mie_fit.fit(image)

    # Calculate final result.
    residual = result.residual.reshape(*dim)
    final = image + residual

    # Write error report.
    report_fit(result)

    ## Make plots.
    # Plot images.
    sns.set(style='white', font_scale=1.4)
    plt.imshow(np.hstack([image, final, residual+1]))
    plt.title('Image, Fit, Residual')
    plt.gray()
    plt.show()

    # Plot Covariance.
    f, ax = plt.subplots()
    #cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.set(font_scale=1.5)
    plt.title('Log Covariance Matrix')
    sns.heatmap(np.log(result.covar), cmap='PuBu',
                square=True, cbar_kws={}, ax=ax)
    ax.set_xticklabels(['x', 'y', 'z', r'a$_p$', r'n$_p$'])
    ax.set_yticklabels([r'n$_p$', r'a$_p$', 'z', 'y', 'x'])
    plt.show()

if __name__ == '__main__':
    example()
