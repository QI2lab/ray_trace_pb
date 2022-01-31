"""
1/25/2022, Peter T. Brown
"""
import numpy as np
import ray_trace as rt
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

wavelength = 0.532
aperture_rad = 2

# O1
n1 = 1.4
na1 = 1.35
alpha1 = np.arcsin(na1 / n1)
mag1 = 100
f1 = 200 / mag1
r1 = na1 * f1

# O2
n2 = 1
na2 = 0.95
alpha2 = np.arcsin(na2 / n2)
mag2 = 40
f2 = 200 / mag2
r2 = na2 * f2

# O3
theta = 30 * np.pi/180
n3 = 1.51
na3 = 1
alpha3 = np.arcsin(na3 / n3)
f3 = 200 / 100
r3 = na3 * f3
o3_normal = np.array([-np.sin(theta), 0, np.cos(theta)])

# tube lens
f_tube_lens_1 = 200
#f_tube_lens_2 = 357
# modified to theoretically correct length...I could see the effect of even a few percent mag error of using 357mm
f_tube_lens_2 = f_tube_lens_1 / f1 *  f2 / n1
f_tube_lens_3 = 200
remote_mag = f_tube_lens_1 / f1 * f2 / f_tube_lens_2 # should = n1 / n2 = 1.4

# positions
p_o1 = n1 * f1 # O1 position
p_t1 = p_o1 + f1 + f_tube_lens_1 # tube lens 1 position
p_t2 = p_t1 + f_tube_lens_1 + f_tube_lens_2 # tube lens 2 position
p_o2 = p_t2 + f_tube_lens_2 + f2 # O2 position
p_remote_focus = p_o2 + n2 * f2
p_o3 = np.array([0, 0, p_remote_focus]) + n3 * f3 * o3_normal
p_pupil_o3 = p_o3 + f3 * o3_normal
p_t3 = p_o3 + (f3 + f_tube_lens_3) * o3_normal
p_imag = p_t3 + f_tube_lens_3 * o3_normal

surfaces = [rt.perfect_lens(f1, [0, 0, p_o1], [0, 0, 1], alpha1),  # O1
            rt.perfect_lens(f_tube_lens_1, [0, 0, p_t1], [0, 0, 1], alpha1),  # tube lens #1
            rt.perfect_lens(f_tube_lens_2, [0, 0, p_t2], [0, 0, 1], alpha2),  # tube lens #2
            rt.perfect_lens(f2, [0, 0, p_o2], [0, 0, 1], alpha2),  # O2
            rt.flat_surface([0, 0, p_remote_focus], o3_normal, r2),
            rt.perfect_lens(f3, p_o3, o3_normal, alpha3),  # O3
            rt.flat_surface(p_pupil_o3, o3_normal, f3*n3), # pupil of 03
            rt.perfect_lens(f_tube_lens_3, p_t3, o3_normal, alpha3),  # tube lens #3
            rt.flat_surface(p_imag, o3_normal, aperture_rad)]
ns = [n1, 1, 1, 1, n2, n3, 1, 1, 1, 1]

# rays
# dx = 10e-3
dx = 0.001
dz = dx * np.tan(theta)
rays = rt.get_ray_fan([dx, 0, dz], alpha1, 101, wavelength=wavelength, nphis=51)
# rays = rt.get_collimated_rays([0, 0, 0], 1, n_disps=31, wavelength=wavelength)
rays = rt.ray_trace_system(rays, surfaces, ns)

# print(np.arctan(rays[0, :, 3] / rays[0, :, 5]) * 180/np.pi)

# plot ray trace surfaces
rt.plot_rays(rays, surfaces)
ax = plt.gca()
ax.axis("equal")

# plot initial angles and remote volume angles
# checking that sine condition is satisfied
# init_angle = np.arctan(rays[0, :, 3] / rays[0, :, 5])
# remote_angle = np.arctan(rays[8, :, 3] / rays[8, :, 5])
#
# figh = plt.figure()
# ax = plt.subplot(1, 1, 1)
# ax.plot(init_angle * 180/np.pi, remote_angle * 180/np.pi)
# ax.plot(init_angle * 180/np.pi, init_angle * 180/np.pi, 'rx')
# ax.set_xlabel("initial angle (deg)")
# ax.set_ylabel("remote angle (deg)")


# plot rays in pupil
rays_pupil = rays[-5]
na = np.array([1, 0, 0])
nc = surfaces[-1].normal
nb = np.cross(nc, na)
c = surfaces[-3].center
x = np.sum((rays_pupil[:, :3] - np.expand_dims(c, axis=0)) * np.expand_dims(na, axis=0), axis=1)
y = np.sum((rays_pupil[:, :3] - np.expand_dims(c, axis=0)) * np.expand_dims(nb, axis=0), axis=1)
phi = rays_pupil[:, -2]

figh = plt.figure()
ax = plt.subplot(1, 1, 1)
im = ax.scatter(x, y, marker='.', c=phi - np.nanmin(phi), cmap="hsv", vmin=0, vmax=np.pi/10)
ax.add_artist(Circle((0, 0), radius=na3*f3, color='k', fill=False))
ax.axis("equal")
plt.colorbar(im)
ax.set_xlabel("position along $n_a$ (mm)")
ax.set_ylabel("position along $n_b$ (mm)")
ax.set_title("Rays and phases in O3 pupil")
ax.set_ylim([-na3*f3, na3*f3])
ax.set_xlim([-na3*f3, na3*f3])