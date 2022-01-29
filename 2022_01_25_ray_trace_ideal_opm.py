"""
1/25/2022, Peter T. Brown
"""
import numpy as np
import ray_trace as rt
import matplotlib.pyplot as plt

wavelength = 0.532
aperture_rad = 2

# O1
n1 = 1.4
na1 = 1.3
f1 = 200 / 100
r1 = na1 * f1

# O2
n2 = 1
na2 = 1
f2 = 200 / 100
r2 = na2 * f2

# O3
theta = 30 * np.pi/180
n3 = 1.5
na3 = 1
f3 = 200 / 100
r3 = na3 * f3
o3_normal = np.array([-np.sin(theta), 0, np.cos(theta)])

# tube lens
f_tube_lens = 200

# positions
p_o1 = n1 * f1
p_t1 = p_o1 + f1 + f_tube_lens
p_t2 = p_t1 + 2 * f_tube_lens
p_o2 = p_t2 + f_tube_lens + f2
p_remote_focus = p_o2 + n2 * f2
p_o3 = np.array([0, 0, p_remote_focus]) + n3 * f3 * o3_normal
p_t3 = p_o3 + (n3 * f3 + f_tube_lens) * o3_normal
p_imag = p_t3 + f_tube_lens * o3_normal

surfaces = [rt.perfect_lens(f1, [0, 0, p_o1], [0, 0, 1], r1),  # O1
            rt.perfect_lens(f_tube_lens, [0, 0, p_t1], [0, 0, 1], r1),  # tube lens #1
            rt.perfect_lens(f_tube_lens, [0, 0, p_t2], [0, 0, 1], r1),  # tube lens #2
            rt.perfect_lens(f2, [0, 0, p_o2], [0, 0, 1], r2),  # O2
            rt.flat_surface([0, 0, p_remote_focus], o3_normal, r2),
            rt.perfect_lens(f3, p_o3, o3_normal, r3),  # O3
            rt.perfect_lens(f_tube_lens, p_t3, o3_normal, r3),  # tube lens #3
            rt.flat_surface(p_imag, o3_normal, aperture_rad)]
ns = [n1, 1, 1, 1, n2, n3, 1, 1, 1]

# rays
rays = rt.get_ray_fan([0, 0, 0], np.arcsin(na1 / n1), 11, wavelength=wavelength)
rays = rt.ray_trace_system(rays, surfaces, ns)

rt.plot_rays(rays, surfaces)
ax = plt.gca()
ax.axis("equal")