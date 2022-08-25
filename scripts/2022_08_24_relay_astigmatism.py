"""
8/24/2022, Peter T. Brown
"""
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import raytrace.raytrace as rt

wavelength = 0.785

# AC508-100-B-ML
# N-LAK22/N-SF6HT
# crown side towards infinity focused/collimated beam
t100c = 13.0
r100c = 65.8
r100i = -56.
t100f = 2.0
r100f = -280.6
bfl100 = 91.5
efl100 = 100

radius = 25.4

surfaces = \
           [# AC508-100-B-ML
            rt.spherical_surface.get_on_axis(r100c, 0, radius),
            rt.spherical_surface.get_on_axis(r100i, t100c, radius),
            rt.spherical_surface.get_on_axis(r100f, t100c + t100f, radius),
            # final focal plane
            rt.flat_surface([0, 0, 110], [0, 0, 1], radius)
            ]

# surface materials
nlak22 = rt.nlak22()
sf6 = rt.sf6()

ns = [1,
      nlak22.n(wavelength), sf6.n(wavelength), 1,
      1]

nrays = 7
rays = np.concatenate((rt.get_collimated_rays([10, 0, 0], 1, nrays, wavelength), # off axis, distributed along x
                       rt.get_collimated_rays([10, 0, 0], 1, nrays, wavelength, phi_start=np.pi/2), # off axis, distributed along y
                       ), axis=0)


rays = rt.ray_trace_system(rays, surfaces, ns)

# plot results
figh, ax = rt.plot_rays(rays[:, :nrays], surfaces, colors="r", figsize=(16, 8))
rt.plot_rays(rays[:, nrays:], surfaces, phi=np.pi/2, colors="b", ax=ax)
plt.show()