"""
Test that perfect lens gives correct phase values
"""
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import raytrace.raytrace as rt

# #########################
# define optical system
# #########################
wavelength = 0.785
aperture = 10
n1 = 1.1
n2 = 1.3
f = 4
na = 1
alpha = np.arcsin(na / n1)

surfaces = [rt.flat_surface([0, 0, 0], [0, 0, 1], aperture),
            rt.perfect_lens(f, [0, 0, n1 * f], [0, 0, 1], alpha),
            rt.flat_surface([0, 0, n1*f + n2*f], [0, 0, 1], aperture)
            ]

materials = [rt.constant(n1),
             rt.constant(n1),
             rt.constant(n2),
             rt.constant(n2)]

# #########################
# define rays
# #########################
nrays = 7
angle = 10 * np.pi/ 180
rays = rt.get_collimated_rays([0, 0, -1], 3, nrays, wavelength,
                              normal=[np.sin(angle), 0, np.cos(angle)])

# #########################
# ray trace system
# #########################
rays_out = rt.ray_trace_system(rays, surfaces, materials)

rt.plot_rays(rays_out, surfaces)

# #########################
# plot phase information
# #########################
sin_t1 = np.sin(np.arctan(rays_out[1, :, 3] / rays_out[1, :, 5]))
h1 = rays_out[1, :, 0]

phi_expected = n1 * 2*np.pi / wavelength * h1 * sin_t1

figh = plt.figure(figsize=(16, 8))

ax = plt.subplot(1, 3, 1)
ax.set_title("phase versus height at FFP")
ax.plot(h1, phi_expected + rays_out[1, nrays//2, 6], 'r')
ax.plot(h1, rays_out[1, :, -2], 'bx')

ax = plt.subplot(1, 3, 2)
ax.set_title("phase versus height at first lens surface")
ax.plot(h1, phi_expected + rays_out[3, nrays//2, 6], 'r')
ax.plot(h1, rays_out[3, :, -2], 'bx')

ax = plt.subplot(1, 3, 3)
ax.set_title("phase versus height at final surface")
ax.plot(h1, rays_out[-1, :, -2], 'bx')

plt.show()