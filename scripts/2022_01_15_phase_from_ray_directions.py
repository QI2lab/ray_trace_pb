"""
Calculate phase profile at a surface from ray directions
"""
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import raytrace.raytrace as rt

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
surfaces = [
    rt.flat_surface([0, 0, 0], [0, 0, 1], aperture_radius),
    # ACT508-200-A-ML
    rt.spherical_surface.get_on_axis(r200f, lens_start, aperture_radius, is_aperture_stop=True),
    rt.spherical_surface.get_on_axis(r200i, lens_start + t200f, aperture_radius),
    rt.spherical_surface.get_on_axis(r200c, lens_start + t200c + t200f, aperture_radius),
    #rt.flat_surface([0, 0, lens_start + t200c + t200f + 0], [0, 0, 1], aperture_radius)
    ]

materials = [rt.constant(1),
             rt.constant(1), rt.sf2(), rt.bk7(), rt.constant(1),
             ]


# auto-focus
surfaces, materials = rt.auto_focus(surfaces, materials, wavelength, mode="paraxial-focused")

abcd = rt.compute_paraxial_matrix(surfaces, [m.n(wavelength) for m in materials], 400)

# ray trace
nrays = 101
rays = rt.get_ray_fan([0, 0, 0], 1*np.pi/180, nrays, wavelength)
rays = rt.ray_trace_system(rays, surfaces, materials)

# plot results
rt.plot_rays(rays, surfaces)

# phases from ray directions compared with explicitely tracked
figh = plt.figure()
dudx = rays[-1, :, 3]
xs = rays[-1, :, 0]
dxs = xs[1:] - xs[:-1]
us = np.cumsum(dudx[:-1] * dxs)
xs_int = 0.5*(xs[1:] + xs[:-1])
k = 2*np.pi/wavelength

plt.plot(rays[-1, :, 0], rays[-1, :, -2], label="opl")
plt.plot(xs_int, k*us + - k * us[nrays//2] + rays[-1, nrays//2, -2], label="ray integral")
plt.legend()
