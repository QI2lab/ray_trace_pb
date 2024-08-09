"""
Aberrations in imaging system formed by two achromats
"""

import numpy as np
import matplotlib.pyplot as plt
import raytrace.raytrace as rt
from raytrace.materials import Nsf11, Ebaf11, Vacuum

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


# l1 = rt.Doublet(Nlak22(),
#                 Nsf6ht(),
#                 radius_crown=65.8,
#                 radius_flint=-280.6,
#                 radius_interface=-56,
#                 thickness_crown=13.0,
#                 thickness_flint=2.0,
#                 aperture_radius=25.4,
#                 input_collimated=False,
#                 names="AC508-100-B"
#                 )
#
# l2 = rt.Doublet(Nlak22(),
#                 Nsf6ht(),
#                 radius_crown=65.8,
#                 radius_flint=-280.6,
#                 radius_interface=-56,
#                 thickness_crown=13.0,
#                 thickness_flint=2.0,
#                 aperture_radius=25.4,
#                 input_collimated=True,
#                 names="AC508-100-B"
#                 )

# flat surface at focal plane of lens 1
cp1 = l1.get_cardinal_points(wlen, Vacuum(), Vacuum())
f1_left = cp1[0]

system = rt.System([rt.FlatSurface([0, 0, 0], [0, 0, 1], 25.4)],
                   []
                   )
system = system.concatenate(l1, Vacuum(), -f1_left[2])

# arrange lenses so collimate
d = l2.find_paraxial_collimated_distance(l2, wlen, Vacuum(), Vacuum(), Vacuum())[0]
system = system.concatenate(l2, Vacuum(), d)
# last surface at focal plane
c2 = l2.get_cardinal_points(wlen, Vacuum(), Vacuum())
d2 = c2[1][2] - l2.surfaces[-1].paraxial_center[2]

system = system.concatenate(rt.FlatSurface([0, 0, 0],
                                           [0, 0, 1],
                                           25.4),
                            Vacuum(),
                            d2)

# ########################
# field curvature
# ########################

nrays = 11
heights = np.linspace(0, 15, 11)
rays = []
for h in heights:
    rays.append(rt.get_ray_fan(np.array([h, 0, 0]),
                               3*np.pi/ 180,
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
system.plot(rays_out, colors='r')

# determine focus position versus height
ints = rt.intersect_rays(rays_out[-1, nrays // 2::nrays],
                  rays_out[-1, nrays//2 + 1::nrays])

# plot field curvature
figh = plt.figure()
ax = figh.add_subplot(1, 1, 1)
ax.plot(heights, ints[:, -1])
ax.set_xlabel("Object position (mm)")
ax.set_ylabel("Focus z-position (mm)")

# ########################
# distortion
# ########################
dxy = 3
nlines = 15
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
ax = figh_dist.add_subplot(1, 1, 1)
ax.plot(rays_grid_out[0, :, 0],
        rays_grid_out[0, :, 1],
        'bx')
ax.plot(rays_grid_out[-1, :, 0],
        rays_grid_out[-1, :, 1],
        'rx')
ax.axis("equal")

plt.show()

