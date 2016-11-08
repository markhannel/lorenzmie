import numpy as np

def hanning(nx, ny):
  """
  Calculates the Hanning Window of size (nx,ny)
  """ 
  if ny <= 0:
    print "Array dimensions must be >= 0"
    return
  if nx <= 0:
    print "Array dimensions must be >= 0"
    return
  row_window = .5*(1-np.cos(2*np.pi*np.arange(0,int(nx))/nx))
  col_window = .5*(1-np.cos(2*np.pi*np.arange(0,int(ny))/ny))
  if ny > 0:
    return np.outer(row_window,col_window)
  else:
    return row_window  

def rayleighsommerfeld(a, z, lamb = 0.447, mpp = 0.135, nozphase = False, 
                       hanning_win = False): 
  """
  Compute electric fields displaced by a distance or set of distances above the
  height z via Rayleigh-Sommerfeld approximation.

  Args:
      a: A two dimensional intensity array.
      z: displacement(s) from the focal plane [pixels].

  Keywords:
      lamb: Wavelength of light in medium [micrometers].
          Default: 0.447
      mpp: Micrometers per pixel.
          Default: 0.135

  Returns:
      Complex electric fields at a plane or set of planes z.
  """

  # Check if a and z are appropriates types
  if type(a) != np.ndarray:
      print 'a must be an numpy array'
      return None

  if type(z) == int:
      z = [z]
  z = np.array(z)
  
  # hologram dimensions
  ndim = a.ndim
  if ndim > 2 : 
      print "requires two-dimensional hologram"
  nx = float(a.shape[1])
  if ndim == 1 : 
      ny = 1.
  else:
      ny = float(a.shape[0])
  
  # volumetric slices  
  if type(z) == int:
    nz = 1
  else:
    nz = len(z)

  # important factors
  ci = complex(0., 1.)
  k = 2.*np.pi*mpp/lamb      # wavenumber in radians/pixels

  # phase factor for Rayleigh-Sommerfeld propagator in Fourier space
  # Refs. [2] and [3]
  qx = np.arange(nx)/nx - 0.5
  qx = ((lamb/mpp) * qx)**2

  # Compute the necessary frequencies
  if ndim == 2 :
    qy = np.arange(ny)/ny - 0.5
    qy = ((lamb/mpp)*qy)**2
    qsq = np.zeros([ny,nx],dtype = complex)
    for i in xrange(0,int(nx)):
      qsq[:,i] += qx[i]
    for j in xrange(0,int(ny)):
      qsq[j,:] += qy[j]
  else:
    qsq = qx

  qfactor = k * np.sqrt(1. - qsq)

  if nozphase:
    qfactor -= k

  if hanning_win:
    qfactor *= hanning(ny, nx)

  ikappa = ci * np.real(qfactor)
  gamma = np.imag(qfactor) 
  
  a = np.array(a,dtype=complex)
  E = np.fft.ifft2(a-1.) # Fourier transform of input field
  E = np.fft.fftshift(E)
  res = np.zeros([ny, nx, nz],dtype = complex)
  for j in xrange(0, nz):
     Hqz = (np.exp((ikappa * z[j] - gamma * abs(z[j]))))
     thisE = E * Hqz                        # convolve with propagator
     thisE = np.fft.ifftshift(thisE)        # Shift Center
     thisE = np.fft.fft2(thisE)             # transform back to real space
     res[:,:,j] = thisE                     # save result

  return res
