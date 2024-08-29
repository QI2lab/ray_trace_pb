import matplotlib.pyplot as plt
import numpy as np
import raytrace.raytrace as rt
from raytrace.materials import Vacuum, Ebaf11, Nsf11

na = 0.98
f = 1
wlen = 0.561

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

# add lenses
system_detect = rt.System([rt.FlatSurface([0, 0, 0],
                                   [0, 0, 1],
                                  f * na),
                    rt.PerfectLens(f,
                                  [0, 0, f],
                                  [0, 0, 1],
                                  np.arcsin(na)),
                    rt.FlatSurface([0, 0, 2*f],
                                   [0, 0, 1],
                                   f * na
                                   ),
                    rt.PerfectLens(f,
                                  [0, 0, 3*f],
                                  [0, 0, 1],
                                  np.arcsin(na)),
                    rt.FlatSurface([0, 0, 4*f],
                                   [0, 0, 1],
                                   f * na
                                   )
                                  ],
                    [Vacuum(), Vacuum(), Vacuum(), Vacuum()]
                                  )

system = system.concatenate(system_detect,
                            Vacuum(),
                            0
                            )

nrays = 101
rays = rt.get_ray_fan([0, 0, 0],
                   10 * np.pi/180,
                   nrays,
                   wlen)
rays_out = system.ray_trace(rays, Vacuum(), Vacuum())

system.plot(rays_out)
plt.show()

figh = plt.figure()
grid = figh.add_gridspec(nrows=1, ncols=1)

ax1 = figh.add_subplot(grid[0])
ax1.set_title('pupil after')
ax1.plot(rays_out[-1, :, 0],
         rays_out[-1, :, 6] - rays_out[-1, nrays // 2, 6],
         label="pupil after")
ax1.plot(rays_out[-9, :, 0],
         rays_out[-9, :, 6] - rays_out[-9, nrays //2 , 6],
         'r.',
         label='pupil before')
ax1.set_xlabel("Height (mm)")
ax1.set_ylabel("OPL (mm)")
ax1.legend()
plt.show()