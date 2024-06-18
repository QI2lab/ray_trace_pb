"""
Test spherical aberration for collimated beam incident on plano-convex lens
"""
import matplotlib.pyplot as plt
import numpy as np
import raytrace.raytrace as rt
from raytrace.materials import Vacuum, Constant

# physical data
wavelength = 0.5 # um
k = 2*np.pi / wavelength

# lens parameters (mm)
aperture_radius = 25.4
t0 = 2.679486355
t1 = 1
rad_curv = 100
n = 1.3
dz = 5

# number of rays to use
nrays = 101 # should be odd

# construct lens
singlet = rt.System([rt.FlatSurface([0, 0, 0], [0, 0, 1], aperture_radius),
                     rt.SphericalSurface.get_on_axis(-rad_curv, t0 + t1, aperture_radius),
                     rt.FlatSurface([0, 0, t0 + t1], [0, 0, 1], aperture_radius)],
                    [Constant(n), Vacuum()])

# generate collimated beam
rays = rt.get_collimated_rays([0, 0, -dz], aperture_radius, nrays, wavelength)

# ray trace
rays = singlet.ray_trace(rays, Vacuum(), Vacuum())
# plot rays
singlet.plot(rays)

# analytic OPL expression
def opl_analytic(h):
    opl =  dz + n * t0 + n * t1 +\
           -n * (rad_curv - np.sqrt(rad_curv**2 - h**2)) + \
           (rad_curv - np.sqrt(rad_curv ** 2 - h ** 2)) / (np.sqrt(1 - n**2 * h**2 / rad_curv**2) * np.sqrt(rad_curv**2 - h**2)/ rad_curv + n * h**2 / rad_curv**2)

    return opl

def opl_quadratic(h):
    opl = dz + n * t0 + n * t1 - (n - 1) * h**2 / 2 / rad_curv
    return opl

def opl_quartic(h):
    opl = opl_quadratic(h) - (n - 1) * h**4 / 8 / rad_curv**3 + (n - 1) **2 * h**4 / 4 / rad_curv**3
    return opl


# plot phases at lens vertex
figh = plt.figure()

ax = figh.add_subplot(1, 2, 1)
ax.plot(rays[0, :, 0], rays[-1, :, -2] / k, 'bx', label="ray trace OPL")
ax.plot(rays[0, :, 0], opl_analytic(rays[0, :, 0]), 'g', label="analytic formula OPL")
ax.set_xlabel("initial ray height (mm)")
ax.set_ylabel("OPL")
ax.set_title("OPL in lens vertex plane versus initial ray height")
ax.legend()

ax = figh.add_subplot(1, 2, 2)
ax.plot(rays[0, :, 0], rays[-1, :, -2] / k - opl_quadratic(rays[0, :, 0]), 'bx', label="ray trace OPL")
ax.plot(rays[0, :, 0], opl_analytic(rays[0, :, 0]) - opl_quadratic(rays[0, :, 0]), 'g', label="analytic formula OPL")
ax.set_xlabel("initial ray height (mm)")
ax.set_ylabel("OPL")
ax.set_title("OPL after subtracting linear and quadratic parts")
