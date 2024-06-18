"""
Calculate phase profile at a Surface from ray directions
"""
import numpy as np
import matplotlib.pyplot as plt
import raytrace.raytrace as rt
from raytrace.materials import Vacuum, Sf2, Bk7

wavelength = 0.532
aperture_radius = 25.4
lens_start = 400
t200c = 10.6
t200f = 6
r200f = 409.4
r200i = 92.1
r200c = -106.2
bfl200 = 190.6
efl200 = 200

system = rt.System([rt.FlatSurface([0, 0, 0], [0, 0, 1], aperture_radius),
                    # ACT508-200-A-ML
                    rt.SphericalSurface.get_on_axis(r200f, lens_start, aperture_radius),
                    rt.SphericalSurface.get_on_axis(r200i, lens_start + t200f, aperture_radius),
                    rt.SphericalSurface.get_on_axis(r200c, lens_start + t200c + t200f, aperture_radius),
                    ],
                   [Vacuum(), Sf2(), Bk7()]
                   )

# auto-focus
focus = system.auto_focus(wavelength, Vacuum(), Vacuum(), mode="paraxial-focused")
system = system.concatenate(rt.FlatSurface(focus, [0, 0, 1], aperture_radius),
                            Vacuum())

# ray trace
nrays = 101
rays = rt.get_ray_fan([0, 0, 0], 1*np.pi/180, nrays, wavelength)
rays = system.ray_trace(rays, Vacuum(), Vacuum())
system.plot(rays)

# phases from ray directions compared with explicitly tracked
figh = plt.figure()
dudx = rays[-1, :, 3]
xs = rays[-1, :, 0]
dxs = xs[1:] - xs[:-1]
us = np.cumsum(dudx[:-1] * dxs)
xs_int = 0.5*(xs[1:] + xs[:-1])
k = 2*np.pi/wavelength

plt.plot(rays[-1, :, 0], rays[-1, :, -2], label="opl")
plt.plot(xs_int,
         k*us + - k * us[nrays//2] + rays[-1, nrays//2, -2],
         label="ray integral")
plt.legend()

plt.show()
