"""
Simulate full SIM/ODT optical setup
"""
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import raytrace.raytrace as rt

wavelength_odt = 0.785
max_height = 25
radius = 25

# #############################
# define lenses and materials used in setup
# #############################
# below c = crown, f = flint, i = intermediate
# radii of curvature are given as if the flint side is first side. If reversed, need to take negatives
# crown side towards infinity space, i.e. is more curved. And BFL is measured from flint side
# (so BFL really defined opposite of the convention I'm using below but ...)
bk7 = rt.bk7()
sf2 = rt.sf2()
sf10 = rt.sf10()
nbaf10 = rt.nbaf10()

# ACT508-200-A-ML
t200c = 10.6
t200f = 6
r200f = 409.4
r200i = 92.1
r200c = -106.2
bfl200 = 190.6
efl200 = 200
# AC508-200-A-ML
t200c_old = 8.5
t200f_old = 2
r200f_old = 376.3
r200i_old = 93.1
r200c_old = -109.9
bfl200_old = 193.7
efl200_old = 200
# AC508-100-A-ML
t100c = 16
t100f = 4
r100f = 363.1
r100i = 44.2
r100c = -71.1
bfl100 = 89
efl100 = 100
# AC508-400-A-ML
t400c = 8
t400f = 8
r400f = 398.5
r400i = 148.9
r400c = -292.3
bfl400 = 396.1
efl400 = 400
# AC508-300-A-ML
t300c = 6.0
t300f = 2.0
r300f = 580.8
r300i = 134
r300c = -161.5
bfl300 = 295.4
efl300 = 300
# olympus 100x NA=1.3 oil-immersion objective (perfect lens)
f_excitation = 1.8
na_excitation = 1.3
fov_excitation = 0.130
# olympus 60x NA=1.0 water-immersion objective (perfect lens) (formerly mitutoyo 50x NA 0.55)
f_detection = 4
na_detection = 0.55

# #############################
# distances for all imaging systems
# #############################
d_dmd_lens = bfl200
d_400_300 = bfl400 + bfl300 + 5
d_300_obj = 300 + 1.8

# only used for SIM imaging
d_200_400 = 200 + 400

# only used for DMD in BFP
d_100_200 = 200 + bfl100
d_100_400 = 100 + 400 - 6

# #############################
# Imaging system for ODT = DMD in BFP
# #############################
l23_shift = 0

l1s_odt = d_dmd_lens
l1e_odt = l1s_odt + t200c + t200f

l2s_odt = l1e_odt + d_100_200 + l23_shift
l2e_odt = l2s_odt + t100c + t100f

l3s_odt = l2e_odt + d_100_400
l3e_odt = l3s_odt + t400c + t400f

l4s_odt = l3e_odt + d_400_300 - l23_shift
l4e_odt = l4s_odt + t300c + t300f

l5s_odt = l4e_odt + d_300_obj
l5e_odt = l5s_odt

n_oil = 1.5
thickness_coverslip = 0.13
n_coverglass = 1.5

focal_plane = l5e_odt + n_oil * f_excitation

n_sample = 1.333
water_thickness = 1

thickness_top_coverslip = 1

water_focal_shift = 0.1275 * 2
l6s_odt = l5e_odt + 1.5 * 1.8 + 4 + water_focal_shift # for 1.5mm water
l6e_odt = l6s_odt

# keep position fixed regardless of water focal shift
l7s_odt = l6s_odt + (f_detection + efl200_old - water_focal_shift)
l7s_odt = l7s_odt + t200c + t200f

camera_pos = l7s_odt + bfl200_old + 11.2

l1 = rt.system([# ACT508-200-A-ML
                rt.spherical_surface.get_on_axis(r200f, 0, radius),
                rt.spherical_surface.get_on_axis(r200i, t200f, radius),
                rt.spherical_surface.get_on_axis(r200c, t200c + t200f, radius)
                ],
                [sf2.n(wavelength_odt), bk7.n(wavelength_odt)]
                )

l2 = rt.system([# AC508-100-A-ML, seems like should have put other way in system?
                rt.spherical_surface.get_on_axis(r100f, 0, radius),
                rt.spherical_surface.get_on_axis(r100i, t100f, radius),
                rt.spherical_surface.get_on_axis(r100c, t100c + t100f, radius)],
                [sf10.n(wavelength_odt), nbaf10.n(wavelength_odt)])

l3 = rt.system([# AC508-400-A-ML
                rt.spherical_surface.get_on_axis(-r400c, 0, radius),
                rt.spherical_surface.get_on_axis(-r400i, t400c, radius),
                rt.spherical_surface.get_on_axis(-r400f, t400c + t400f, radius)
                ],
                [bk7.n(wavelength_odt), sf2.n(wavelength_odt)])

l4 = rt.system([# AC508-300-A-ML
                rt.spherical_surface.get_on_axis(r300f, 0, radius),
                rt.spherical_surface.get_on_axis(r300i, t300f, radius),
                rt.spherical_surface.get_on_axis(r300c, t300c + t300f, radius)
                ],
                [sf2.n(wavelength_odt), bk7.n(wavelength_odt)])



comb = l1.concatenate(l2, rt.vacuum(), d_100_200)



surfaces_odt =  \
               [# ACT508-200-A-ML
                rt.spherical_surface.get_on_axis(r200f, l1s_odt, radius),
                rt.spherical_surface.get_on_axis(r200i, l1s_odt + t200f, radius),
                rt.spherical_surface.get_on_axis(r200c, l1s_odt + t200c + t200f, radius),
                # AC508-100-A-ML, seems like should have put other way in system?
                rt.spherical_surface.get_on_axis(r100f, l2s_odt, radius),
                rt.spherical_surface.get_on_axis(r100i, l2s_odt + t100f, radius),
                rt.spherical_surface.get_on_axis(r100c, l2s_odt + t100c + t100f, radius),
                # AC508-400-A-ML
                rt.spherical_surface.get_on_axis(-r400c, l3s_odt, radius),
                rt.spherical_surface.get_on_axis(-r400i, l3s_odt + t400c, radius),
                rt.spherical_surface.get_on_axis(-r400f, l3s_odt + t400c + t400f, radius),
                # AC508-300-A-ML
                rt.spherical_surface.get_on_axis(r300f, l4s_odt, radius),
                rt.spherical_surface.get_on_axis(r300i, l4s_odt + t300f, radius),
                rt.spherical_surface.get_on_axis(r300c, l4s_odt + t300c + t300f, radius),
                # excitation objective
                rt.perfect_lens(f_excitation, [0, 0, l5s_odt], [0, 0, 1], f_excitation * na_excitation),
                # oil
                rt.flat_surface([0, 0, focal_plane - thickness_coverslip], [0, 0, 1], radius),
                # coverslip (assume focal plane right at coverslip)
                rt.flat_surface([0, 0, focal_plane], [0, 0, 1], fov_excitation),
                # water/sample
                rt.flat_surface([0, 0, focal_plane + water_thickness], [0, 0, 1], radius),
                # top cover-glass
                rt.flat_surface([0, 0, focal_plane + water_thickness + thickness_top_coverslip], [0, 0, 1], radius),
                # detection objective
                rt.perfect_lens(f_detection, [0, 0, l6s_odt], [0, 0, 1], f_detection * na_detection),
                # ACT08-200-A-ML
                rt.spherical_surface.get_on_axis(-r200c_old, l7s_odt, radius),
                rt.spherical_surface.get_on_axis(-r200i_old, l7s_odt + t200c_old, radius),
                rt.spherical_surface.get_on_axis(-r200f_old, l7s_odt + t200c_old + t200f_old, radius),
                # final focal plane
                rt.flat_surface([0, 0, camera_pos], [0, 0, 1], radius)
                ]

materials = [rt.constant(1),
             rt.sf2(), rt.bk7(), rt.constant(1), #ACT508-200-A-ML
             rt.sf10(), rt.nbaf10(), rt.constant(1), # AC508-100-A-ML
             rt.bk7(), rt.sf2(), rt.constant(1), # AC508-400-A-ML
             rt.sf2(), rt.bk7(), rt.constant(1), # AC508-300-A-ML
             rt.constant(n_oil), # immersion oil
             rt.constant(n_coverglass), # coverslip
             rt.constant(n_sample), # sample
             rt.constant(n_coverglass), # top cover-glass
             rt.constant(1), # air before detection objective
             rt.constant(1), # air after detection objective
             rt.bk7(), rt.sf2(), rt.constant(1), # ACT08-200-A-ML
             rt.constant(1) # air before final focal plane
             ]

# #######################################
# ray tracing
# #######################################
max_angle = 0.5 * np.pi/180
sep = 4 * 0.55 * (1.8/4 * 400/300 * 200/100) # edge of mitutoyo pupil
lateral_shift = 0 * -np.sin(20 * np.pi/180) * sep
nrays = 7
rays = np.concatenate((rt.get_ray_fan([0, 0, 0], max_angle, nrays, wavelength_odt),
                       rt.get_ray_fan([1/3 * sep, 0, 1/3 * lateral_shift], max_angle, nrays, wavelength_odt),
                       rt.get_ray_fan([2/3 * sep, 0, 2/3 * lateral_shift], max_angle, nrays, wavelength_odt),
                       rt.get_ray_fan([sep, 0, lateral_shift], max_angle, nrays, wavelength_odt)),
                      axis=0)

rays = rt.ray_trace_system(rays, surfaces_odt, materials)

# #######################################
# plot results
# #######################################
figh = rt.plot_rays(rays, surfaces_odt, colors=["k"] * nrays + ["b"] * nrays + ["r"] * nrays + ["g"] * nrays, figsize=(16, 8))
ax = plt.gca()
ax.plot([l5s_odt - 1.8, l5s_odt - 1.8], [-10, 10], 'k--')
plt.suptitle("ODT")
