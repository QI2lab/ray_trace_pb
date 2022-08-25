"""
Test that perfect lens gives correct phase values
"""
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import raytrace.raytrace as rt

wavelength = 0.785
k = 2*np.pi/ wavelength
n1 = 1.1
n2 = 1.3
f = 4
nrays = 7
rays = rt.get_collimated_rays([0, 0, -1], 3, nrays, wavelength, normal=[np.sin(10 * np.pi/180), 0, np.cos(10*np.pi/180)])
# rays[-1, -2] -= 1

surfaces = [rt.flat_surface([0, 0, 0], [0, 0, 1], 10),
            rt.perfect_lens(f, [0, 0, n1 * f], [0, 0, 1], 10),
            rt.flat_surface([0, 0, n1*f + n2*f], [0, 0, 1], 10)
            ]
ns = [n1, n1, n2, n2]

rays_out = rt.ray_trace_system(rays, surfaces, ns)

rt.plot_rays(rays_out, surfaces)

figh = plt.figure(figsize=(16, 8))

sin_t1 = rays_out[1, :, 3] / rays_out[1, :, 5]
h1 = rays_out[1, :, 0]

ax = plt.subplot(1, 3, 1)
ax.set_title("phase versus height at FFP")
phi_off = rays_out[1, nrays//2, 6]
ax.plot(h1, k * h1 * n1 * sin_t1 + phi_off, 'r')
ax.plot(h1, rays_out[1, :, -2], 'bx')

ax = plt.subplot(1, 3, 2)
ax.set_title("phase versus height at first lens surface")
phi_off = rays_out[3, nrays//2, 6]
ax.plot(h1, k * h1 * n1 * sin_t1 + phi_off, 'r')
ax.plot(h1, rays_out[3, :, -2], 'bx')

ax = plt.subplot(1, 3, 3)
ax.set_title("phase versus height at final surface")
ax.plot(h1, rays_out[-1, :, -2], 'bx')

plt.show()