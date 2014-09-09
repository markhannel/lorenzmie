#+
# NAME:
#       spherefield
#
# PURPOSE:
#       Calculates the complex electric field scattered by a sphere illuminated
#       by a plane wave linearly polarized in the x direction.
#
# CATEGORY:
#       Holography, light scattering, microscopy
#
# CALLING SEQUENCE:
#       field = spherefield(x, y, z, a, np, 
#                           nm = nm, lamb = lamb, 
#                           mpp = mpp, k = k)
#
# INPUTS:
#       x: [npts] array of pixel coordinates [pixels]
#       y: [npts] array of pixel coordinates [pixels]
#       z: If field is required in a single plane, 
#          z is the plane's distance from the sphere's center
#          [pixels].
#          Otherwise, z is an [npts] array of coordinates.
#
#       NOTE: Ideally, x, y and z should be double precision.
#             This is left to the calling program for efficiency.
#
#       a: radius of sphere [micrometers]
#
#       np : (complex) refractive index of sphere
#
# KEYWORD PARAMETERS:
#       nm: (complex) refractive index of medium.
#           Default: 1.33 (water)
#
#       lamb: vacuum wavelength of light [micrometers]
#           Default: 0.632 (HeNe)
#
#       mpp: Microns per pixel.  
#           Default: 0.135 (Nikon rig)
#
#       precision: precision with which field is calculated.
#           Example: precision=1.e-6 allows errors smaller than
#           one part in a million.
#           Default: Machine precision (effectively)
#
# KEYWORD FLAGS:
#       cartesian: If set, return field components in Cartesian
#            coordinates.  Default: spherical polar coordinates
#
#       gpu: If set, use GPU_SPHERICALFIELD to
#            compute series of vector spherical harmonics.
#            This can substantially improve performance on systems
#            with GPUlib installed.
#
# OUTPUTS:
#       field: [3, npts] complex values of field at the positions r.
#              [0, :] r component
#              [1, :] theta component
#              [2, :] phi component
#
#              If CARTESIAN is set:
#              [0, :] x component (incident polarization)
#              [1, :] y component (transverse component)
#              [2, :] z component (axial component, relative to
#              incident beam).
#
# PROCEDURE:
#   Calls SPHERE_COEFFCIENTS to obtain scattering
#   coefficients, and then calls SPHERICALFIELD to sum up
#   the series of vector spherical harmonics.
#
# REFERENCE:
#   1. Adapted from Chapter 4 in
#      C. F. Bohren and D. R. Huffman, 
#      Absorption and Scattering of Light by Small Particles, 
#      (New York, Wiley, 1983).
# 
# MODIFICATION HISTORY:
# Written by David G. Grier, New York University, 5/2007
# 06/09/2007: DGG finally read Section 4.8 in Bohren and Huffman about
#    numerical stability of the recursions used to compute the scattering
#    coefficients.  Feh.  Result is a total rewrite.
# 06/20/2007: DGG Calculate \tau_n(\cos\theta) and \pi_n(\cos\theta)
#    according to recurrence relations in 
#    W. J. Wiscombe, Appl. Opt. 19, 1505-1509 (1980).
#    This is supposed to improve numerical accuracy.
# 02/08/2008: DGG. Replaced single [3, npts] array of input coordinates
#    with two [npts] arrays for x and y, and a separate input for z.
#    Eliminated):uble() call for coordinates.  Z may have 1 element or
#    npts elements. Small documentation fixes.
# 04/03/2008: Bo Sun (Sephiroth), NYU: Calculate Lorenz-Mie a and b
#    coefficients using continued fractions rather than recursion.
#    Osman Akcakir from Arryx pointed out that the results are
#    more accurate in extreme cases.  Method described in
#    William J. Lentz, "Generating Bessel functions in Mie scattering
#    calculations using continued fractions, " Appl. Opt. 15, 668-671
#    (1976).
# 04/04/2008: DGG small code clean-ups and documentation.  Added
#    RECURSIVE keyword for backward compatibility in computing a and b
#    coefficients.
# 04/11/2008: Sephiroth: Corrected small error in jump code for
#    repeated fractions in Mie coefficients.
# 06/25/2008: DGG Don't clobber x coordinate input values.
# 10/09/2008: DGG converted to a shell for calling SPHERE_COEFFICIENTS
#    and SPHERICALFIELD.  This greatly simplifies generalizing the
#    code to other systems.  Added PRECISION keyword.
# 10/12/2008: DGG Added GPU keyword.
# 01/15/2009: DGG check for PRECISION with KEYWORD_SET to avoid
#    unnecessary processing when PRECISION=0.  Documentation cleanups.
# 06/18/2010: DGG Added COMPILE_OPT.
# 04/06/2011 DGG Default to double precision on GPU.
#
# Copyright (c) 2006-2011, Bo Sun and David G. Grier.
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

### Import Libraries
#
from numpy import *
from sphericalfield import *
from sphere_coefficients import *


def spherefield(x, y, z, a, np,nm = complex(1.3326, 1.5e-8), 
                lamb = 0.6328, mpp = 0.101, precision = '',
                cartesian = '', gpu = ''):
   """
   Calculates the complex electric field scattered by a sphere illuminated
   by a plane wave linearly polarized in the x direction.
   
   Inputs:
   x:  [npts] array of pixel coordinates [pixels]
   y:  [npts] array of pixel coordinates [pixels]
   z:  If field is required in a single plane, 
       z is the plane's distance from the sphere's center
       [pixels].
       Otherwise, z is an [npts] array of coordinates.
   a:  radius of sphere [micrometers]
   np: (complex) refractive index of sphere

   Parameters:
   nm:         refractive index of medium (default water)
   lamb:       vacuum wavelength of light (default 0.6328)
   mpp:        micrometers per pixel      (default 0.101)
   precision:  accuracy of scattering coefficients
   k:          scaled wavenumber in medium
   cartesian:  project to cartesian coordinates
   gpu:        use GPU acceleration

   Example:
   (Using default parameters)
   >>> ComplexE = spherefield(x,y,z,a,np)
   """
   ab = sphere_coefficients(a, np, nm, lamb)

   if type(precision) != str : 
      fac = abs(ab[:, 1])
      w = nmp.where(fac > precision*max(fac))
      w = nmp.concatenate((array([0]),w[0])) # retain first coefficient for bookkeeping
      ab =  ab[w,:]
 
   lamb_m = lamb / real(nm) / mpp # medium wavelength [pixel]

   k = 2.0 * nmp.pi / lamb_m          # wavenumber in medium [pixel**{-1}]

   if gpu == 'gpu' :
      field = gpu_sphericalfield(x, y, z, ab, lamb_m, cartesian=cartesian)
   else:
      field = sphericalfield(x, y, z, ab, lamb_m, cartesian=cartesian)

   return field

