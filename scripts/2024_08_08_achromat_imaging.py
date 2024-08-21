"""
Aberrations in imaging system formed by two achromats
"""

import numpy as np
import matplotlib.pyplot as plt
import raytrace.raytrace as rt
from raytrace.materials import Nsf11, Ebaf11, Vacuum
from localize_psf.fit import fit_model

wlen = 0.635

l1 = rt.Doublet(Ebaf11(),
                Nsf11(),
                radius_crown=50.8,
                radius_flint=-247.7,
                radius_interface=-41.7,
                thickness_crown=20.,
                thickness_flint=3.,
                aperture_radius=25.4,
                input_collimated=False,
                names="AC508-075-A-ML")

l2 = rt.Doublet(Ebaf11(),
                Nsf11(),
                radius_crown=50.8,
                radius_flint=-247.7,
                radius_interface=-41.7,
                thickness_crown=20.,
                thickness_flint=3.,
                aperture_radius=25.4,
                input_collimated=True,
                names="AC508-075-A-ML")

# flat surface at focal plane of lens 1
cp1 = l1.get_cardinal_points(wlen, Vacuum(), Vacuum())
f1_left = cp1[0][-1]
f1_right = cp1[1][-1]
wd_right = f1_right - l1.surfaces[-1].paraxial_center[-1]

system = rt.System([rt.FlatSurface([0, 0, 0], [0, 0, 1], 25.4)],
                   []
                   )
system = system.concatenate(l1, Vacuum(), -f1_left)

# find collimated distance between l1 and l2
d = l2.find_paraxial_collimated_distance(l2, wlen, Vacuum(), Vacuum(), Vacuum())

# add flat surface at Fourier plane
system = system.concatenate(rt.FlatSurface([0, 0, 0],
                                           [0, 0, 1],
                                           25.4),
                            Vacuum(),
                            wd_right)
ind_pupil = len(system.surfaces) - 1

# add lens #2
system = system.concatenate(l2, Vacuum(), d - wd_right)


# last surface at focal plane
c2 = l2.get_cardinal_points(wlen, Vacuum(), Vacuum())
wd2 = c2[1][2] - l2.surfaces[-1].paraxial_center[2]

system = system.concatenate(rt.FlatSurface([0, 0, 0],
                                           [0, 0, 1],
                                           25.4),
                            Vacuum(),
                            wd2)
system.set_aperture_stop(ind_pupil)

# ########################
# gaussian beam analysis
# ########################
# q_in = gb.get_q(10e-3, 0, wlen, 0, 1)
# qs = system.gaussian_paraxial(q_in,
#                               wlen,
#                               Vacuum(),
#                               Vacuum(),
#                               print_results=True)

# ########################
# seidel analysis
# ########################
abs = system.seidel_third_order(wlen,
                                Vacuum(),
                                Vacuum(),
                                print_results=True,
                                object_height=5
                                )


# ########################
# field curvature
# ########################
nrays = 5
heights = np.linspace(0, 16, 21, endpoint=True)
rays = []
for h in heights:
    rays.append(rt.get_ray_fan(np.array([h, 0, 0]),
                               1*np.pi/ 180,
                               nrays,
                               wlen
                               )
                )
rays = np.concatenate(rays, axis=0)

# ray trace
rays_out = system.ray_trace(rays,
                            Vacuum(),
                            Vacuum(),
                            )
figh_curv, _ = system.plot(rays_out, colors='r')
figh_curv.suptitle("Field curvature")

# determine focus position versus height
ints = rt.intersect_rays(rays_out[-1, nrays // 2::nrays],
                  rays_out[-1, nrays//2 + 1::nrays])
d_shift = ints[:, -1] - ints[0, -1]

def quad(h, p):
    return p[0] - 0.5*h**2 / p[1]

result = fit_model(d_shift, lambda p: quad(heights, p),
                   init_params=[0, 10])

h_interp = np.linspace(0, np.max(heights), 300, endpoint=True)
d_fit = quad(h_interp, result["fit_params"])


# plot field curvature
figh = plt.figure()
figh.suptitle(f"R Petzval = {result['fit_params'][1]:.1f}mm")
ax = figh.add_subplot(1, 1, 1)
ax.plot(h_interp, d_fit, 'r')
ax.plot(heights, d_shift, 'rx')
ax.set_xlabel("Object height (mm)")
ax.set_ylabel("Focus z-position shift (mm)")
ax.axis("equal")

# ########################
# spherical aberration
# ########################
nrays_sph = 101
rays_sph = rt.get_ray_fan([0, 0, 0], 20 * np.pi/180, nrays_sph, wlen)
rays_sph_out = system.ray_trace(rays_sph, Vacuum(), Vacuum())

figh_sph_rt, _ = system.plot(rays_sph_out, colors="r")
figh_sph_rt.suptitle("Spherical aberration")

angle = np.arctan2(rays_sph_out[0, :, 3], rays_sph_out[0, :, 5])
axis_intersection = rt.intersect_rays(rays_sph_out[0, nrays_sph // 2],
                                      rays_sph_out[-1])

figh_sph = plt.figure()
ax = figh_sph.add_subplot(1, 1, 1)
figh_sph.suptitle("Spherical aberration")
ax.plot(angle * 180/np.pi, axis_intersection[:, -1], 'rx')
ax.set_xlabel("Object space angle (deg)")
ax.set_ylabel("Axis crossing z-position (mm)")
ax.set_xlim([None, 0])

# ########################
# distortion
# ########################
dxy = 3
nlines = 16
nrays_line = nlines * 11
rays_grid = []
for ii in range(nlines):
    vline = np.zeros((nrays_line, 8))
    vline[:, 0] = dxy * (ii - nlines // 2)
    vline[:, 1] = np.linspace(-dxy * (nlines//2),
                              dxy * (nlines//2),
                              nrays_line,
                              endpoint=True)
    vline[:, 5] = 1
    vline[:, 7]= wlen

    hline = np.zeros((nrays_line, 8))
    hline[:, 0] = vline[:, 1]
    hline[:, 1] = vline[:, 0]
    hline[:, 5] = 1
    hline[:, 7] = wlen

    rays_grid.append(vline)
    rays_grid.append(hline)
rays_grid = np.concatenate(rays_grid, axis=0)

rays_grid_out = system.ray_trace(rays_grid, Vacuum(), Vacuum())

figh_dist = plt.figure()
figh_dist.suptitle("distortion")
ax = figh_dist.add_subplot(1, 2, 1)
ax.plot(rays_grid_out[0, :, 0],
        rays_grid_out[0, :, 1],
        'bx',
        label="object")
ax.plot(rays_grid_out[-1, :, 0],
        rays_grid_out[-1, :, 1],
        'rx',
        label="image")
ax.axis("equal")
ax.legend()
ax.set_xlabel("x-position (mm)")
ax.set_ylabel("y-position (mm)")

ax = figh_dist.add_subplot(1, 2, 2)
rin = np.linalg.norm(rays_grid_out[0, :, (0, 1)], axis=0)
rout = np.linalg.norm(rays_grid_out[-1, :, (0, 1)], axis=0)
ax.axvline(0, c='gray')
ax.axhline(0, c='gray')
ax.plot([0, 20], [0, 20], 'k')
ax.plot(rin, rout, 'r')
ax.set_xlabel("rho start (mm)")
ax.set_ylabel("rho end (mm)")

plt.show()

