#+
# NAME:
#    spheredhm
#
# PURPOSE:
#    Computes holographic microscopy image of a sphere
#    immersed in a transparent medium.
#
# CATEGORY:
#    Holographic microscopy
#
# CALLING SEQUENCE:
#    holo = spheredhm(rp, ap, np, nm, dim)
#
# INPUTS:
#    rp  : [x, y, z] 3 dimensional position of sphere relative to center
#          of image.
#    ap  : radius of sphere [micrometers]
#    np  : (complex) refractive index of sphere
#    nm  : (complex) refractive index of medium
#    dim : [nx, ny] dimensions of image [pixels]
#
#    NOTE: The imaginary parts of the complex refractive indexes
#    should be positive for absorbing materials.  This follows the
#    convention used in SPHERE_COEFFICIENTS.
#
# OUTPUTS:
#    holo: [nx, ny] real-valued digital holographic image of sphere.
#
# KEYWORDS:
#    alpha: fraction of incident light scattered by particle.
#           Default: 1.
#
#    lamb: vacuum wavelength of light [micrometers]
#
#    mpp: micrometers per pixel
#
#    precision: relative precision with which fields are calculated.
#
# KEYWORD FLAGS:
#    gpu: Use GPU accelerated calculation.  This only works on
#         systems with GPUlib installed.  Requires NVIDIA graphics
#         card with CUDA installed.
#
#    lut: interpolate two-dimensional result from one-dimensional 
#         look-up table, with _substantial_ speed benefits, at the
#         cost of some precision.
#
# PROCEDURE:
#    Calls SPHEREFIELD to compute the field.
#
# REFERENCE:
#    S. Lee, Y. Roichman, G. Yi, S. Kim, S. Yang, A. van Blaaderen, 
#    P. van Oostrum and D. G. Grier, 
#    Chararacterizing and tracking single colloidal particles with 
#    video holographic microscopy, 
#    Optics Express 15, 18275-18282 (2007)

# EXAMPLE:
#    Display a DHM image of a 1.5 micrometer diameter polystyrene
#    sphere (np = 1.5) in water (nm = 1.33).
#
#    IDL> tvscl, spheredhm([0, 0, 200], 0.75, 1.5, 1.33, [201, 201])
#
# MODIFICATION HISTORY:
#  Written by David G. Grier, New York University, 3/2007
#  05/25/2007 DGG: Added ALPHA keyword.
#  02/08/2008 DGG: Adopted updated SPHEREFIELD syntax:
#             separate x, y, and z coordinates.
#  10/09/2008 DGG: Added LAMBDA and PRECISION keywords
#  10/14/2008 DGG: Added GPU keyword.
#  10/16/2008 DGG: Added LUT keyword.
#  03/15/2010 DGG: Documentation cleanups.
#  11/03/2010 DGG: ALPHA defaults to 1.  Clean up image calculation.
#    Documentation fixes and formatting.
#  06/23/2012 DGG flipped sign of xp and yp for consistency with
#    fitspheredhm and dggdhmspheredhm.
#
# Copyright (c) 2007-2012 David G. Grier
#
# UPDATES:
#    The most recent version of this program may be obtained from
#    http://physics.nyu.edu/grierlab/software.html
# 
# LICENSE:
#    This program is free software# you can redistribute it and/or
#    modify it under the terms of the GNU General Public License as
#    published by the Free Software Foundation# either version 2 of the
#    License, or (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful, 
#    but WITHOUT ANY WARRANTY# without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program# if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
#    02111-1307 USA
#
#    If the Internet and WWW are still functional when you are using
#    this, you should be able to access the GPL here: 
#    http://www.gnu.org/copyleft/gpl.html
#-

###
# Import Libraries
#
from numpy import *
from spherefield import *

def spheredhm(rp, ap, np, nm, dim, 
              alpha = '', 
              mpp = 0.101,
              lamb = .6328,
              precision = '', 
              gpu = '', 
              lut = ''):
    """
    Computes holographic microscopy image of a sphere
    immersed in a transparent medium.
    
    holo = spheredhm(rp, ap, np, nm, dim)

    INPUTS:
    rp  : [x, y, z] 3 dimensional position of sphere relative to center
          of image.
    ap  : radius of sphere [micrometers]
    np  : (complex) refractive index of sphere
    nm  : (complex) refractive index of medium
    dim : [nx, ny] dimensions of image [pixels]

    NOTE: The imaginary parts of the complex refractive indexes
    should be positive for absorbing materials.  This follows the
    convention used in SPHERE_COEFFICIENTS.

    Parameters:
    precision: 
    alpha: fraction of incident light scattered by particle.
           Default: 1.
    lamb:  vacuum wavelength of light [micrometers]
    mpp: micrometers per pixel
    precision: relative precision with which fields are calculated.

    Example:
    >>> image = spheredhm([0.,0.,200.],0.75,1.5,1.33,[201,201])
    """



    nx = float(dim[0])
    ny = float(dim[1])
    npts = nx * ny
    x = arange(npts) % nx
    y = floor(arange(npts) / nx)
    x -= nx/2. + float(rp[0])
    y -= ny/2. + float(rp[1])

    zp = float(rp[2])

    if lut == 'lut':
        rho = sqrt(x**2 + y**2)
        x = arange(fix(rho).max()+1)
        y = 0. * x
    
    k = 2.0*pi/(lamb/real(nm)/mpp)
        
    field = spherefield(x, y, zp, ap, np, nm = nm, 
                        cartesian='cartesian', mpp = mpp, 
                        lamb = lamb, precision = precision, 
                        gpu = gpu)

    if len(alpha) == 1 : 
       field *= alpha

    # scattered intensity
    dhm = sum(real(field * conj(field)), 1)
    # interference between scattered wave and incident plane wave
    dhm += 2.0 * real(field[:, 0] * exp(complex(0, -k*zp)))
    # intensity of plane wave
    dhm += 1.0

    if lut == 'lut' : 
      dhm = interpolate(dhm, rho, cubic=-0.5)

    return dhm.reshape(ny,nx)
