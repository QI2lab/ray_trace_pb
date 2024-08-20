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
system = system.concatenate(rt.FlatSurface([0, 0, 0], [0, 0, 1], np.inf),
                            Vacuum(),
                            wd_right)
ind_pupil = len(system.surfaces)

# add lens #2
system = system.concatenate(l2, Vacuum(), d - wd_right)

# set stop
system.set_aperture_stop(ind_pupil)

system.seidel_third_order()
