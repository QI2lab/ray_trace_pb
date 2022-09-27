"""
Simulate full SIM/ODT optical setup
"""
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import raytrace.raytrace as rt

wavelength = 0.785
max_height = 25
radius = 25

# #############################
# define lenses and materials used in setup
# #############################
# below c = crown, f = flint, i = intermediate
# radii of curvature are given as if the flint side is first side. If reversed, need to take negatives
# crown side towards infinity space, i.e. is more curved. And BFL is measured from flint side
# (so BFL really defined opposite of the convention I'm using below but ...)

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
# Imaging system for ODT = DMD in BFP
# #############################
n_coverglass = 1.5
thickness_coverslip = 0.13
n_oil = 1.5
thickness_oil = (f_excitation - thickness_coverslip / n_coverglass) * n_oil # f = d1/n1 + d2/n2

n_sample = 1.333
thickness_sample = 1
thickness_top_coverslip = 1
n_water = 1.333
thickness_water_immersion = (f_detection - thickness_sample / n_sample - thickness_top_coverslip / n_coverglass) * n_water


l1 = rt.system([# ACT508-200-A-ML
                rt.spherical_surface.get_on_axis(r200f, bfl200, radius),
                rt.spherical_surface.get_on_axis(r200i, bfl200 + t200f, radius),
                rt.spherical_surface.get_on_axis(r200c, bfl200 + t200c + t200f, radius)
                ],
                [rt.sf2(), rt.bk7()]
                )

l2 = rt.system([# AC508-100-A-ML, seems like should have put other way in system?
                rt.spherical_surface.get_on_axis(r100f, 0, radius),
                rt.spherical_surface.get_on_axis(r100i, t100f, radius),
                rt.spherical_surface.get_on_axis(r100c, t100c + t100f, radius)],
                [rt.sf10(), rt.nbaf10()])

l3 = rt.system([# AC508-400-A-ML
                rt.spherical_surface.get_on_axis(-r400c, 0, radius),
                rt.spherical_surface.get_on_axis(-r400i, t400c, radius),
                rt.spherical_surface.get_on_axis(-r400f, t400c + t400f, radius)
                ],
                [rt.bk7(), rt.sf2()])

l4 = rt.system([# AC508-300-A-ML
                rt.spherical_surface.get_on_axis(r300f, 0, radius),
                rt.spherical_surface.get_on_axis(r300i, t300f, radius),
                rt.spherical_surface.get_on_axis(r300c, t300c + t300f, radius)
                ],
                [rt.sf2(), rt.bk7()])

obj1 = rt.system([rt.perfect_lens(f_excitation, [0, 0, 0], [0, 0, 1], f_excitation * na_excitation)],
                 [])

sample = rt.system([# oil
                    rt.flat_surface([0, 0, 0], [0, 0, 1], radius),
                    # coverslip (assume focal plane right at coverslip)
                    rt.flat_surface([0, 0, thickness_coverslip], [0, 0, 1], fov_excitation),
                    # water/sample
                    rt.flat_surface([0, 0, thickness_coverslip + thickness_sample], [0, 0, 1], radius),
                    # top cover-glass
                    rt.flat_surface([0, 0, thickness_coverslip + thickness_sample + thickness_top_coverslip], [0, 0, 1], radius)
                    ],
                    [rt.constant(n_coverglass), # coverslip
                    rt.constant(n_sample), # sample
                    rt.constant(n_coverglass)])

obj2 = rt.system([rt.perfect_lens(f_detection, [0, 0, 0], [0, 0, 1], f_detection * na_detection)],
                 [])

l8 = rt.system([ # ACT08-200-A-ML
                rt.spherical_surface.get_on_axis(-r200c_old, 0, radius),
                rt.spherical_surface.get_on_axis(-r200i_old, t200c_old, radius),
                rt.spherical_surface.get_on_axis(-r200f_old, t200c_old + t200f_old, radius)
                ],
                [rt.bk7(), rt.sf2()])


# compute working distances and other paraxial info about lenses
_, fp1_b, _, _, efl1, _ = l1.get_cardinal_points(wavelength, rt.vacuum(), rt.vacuum())
wd1_right = (fp1_b - l1.surfaces[-1].paraxial_center)[2]

fp2_a, fp2_b, _, _, efl2, _ = l2.get_cardinal_points(wavelength, rt.vacuum(), rt.vacuum())
wd2_left = (l2.surfaces[0].paraxial_center - fp2_a)[2]
wd2_right = (fp2_b - l2.surfaces[-1].paraxial_center)[2]

fp3_a, fp3_b, _, _, efl3, _ = l3.get_cardinal_points(wavelength, rt.vacuum(), rt.vacuum())
wd3_left = (l3.surfaces[0].paraxial_center - fp3_a)[2]
wd3_right = (fp3_b - l3.surfaces[-1].paraxial_center)[2]

fp4_a, fp4_b, _, _, efl4, _ = l4.get_cardinal_points(wavelength, rt.vacuum(), rt.vacuum())
wd4_left = (l4.surfaces[0].paraxial_center - fp4_a)[2]
wd4_right = (fp4_b - l4.surfaces[-1].paraxial_center)[2]

fp5_a, fp5_b, _, _, efl5, _ = obj1.get_cardinal_points(wavelength, rt.vacuum(), rt.vacuum())
wd5_left = (obj1.surfaces[0].paraxial_center - fp5_a)[2]
# wd5_right = (fp5_b - obj1.surfaces[-1].paraxial_center)[2]

fp7_a, fp7_b, _, _, efl7, _ = obj2.get_cardinal_points(wavelength, rt.vacuum(), rt.vacuum())
# wd7_left = (obj2.surfaces[0].paraxial_center - fp7_a)[2]
wd7_right = (fp7_b - obj2.surfaces[-1].paraxial_center)[2]

fp8_a, fp8_b, _, _, efl8, _ = l8.get_cardinal_points(wavelength, rt.vacuum(), rt.vacuum())
wd8_left = (l8.surfaces[0].paraxial_center - fp8_a)[2]
wd8_right = (fp8_b - l8.surfaces[-1].paraxial_center)[2]

# create optical system using paraxial working distances to set spacing between lenses
ls = l1.concatenate(l2, rt.vacuum(), wd1_right + wd2_left)
ls = ls.concatenate(l3, rt.vacuum(), wd2_right + wd3_left)
ls = ls.concatenate(l4, rt.vacuum(), wd3_right + wd4_left)
ls = ls.concatenate(obj1, rt.vacuum(), wd4_right + wd5_left)
ls = ls.concatenate(sample, rt.constant(n_oil), thickness_oil)
ls = ls.concatenate(obj2, rt.constant(n_water), thickness_water_immersion) # detection objective
ls = ls.concatenate(l8, rt.vacuum(), wd7_right + wd8_left) # tube lens
ls = ls.concatenate(rt.system([rt.flat_surface([0, 0, 0], [0, 0, 1], radius)], []), # ad camera
                    rt.vacuum(), wd8_right)

#abcd = ls.compute_paraxial_matrix(wavelength, rt.vacuum(), rt.vacuum())

# #######################################
# ray tracing
# #######################################
max_angle = 0.5 * np.pi/180
sep = f_detection * na_detection * (f_excitation/f_detection * 400/300 * 200/100) # edge of mitutoyo pupil
lateral_shift = -np.sin(20 * np.pi/180) * sep
pupil_fractions = [0, 1/3, 2/3, 1]
nrays = 11

rays = np.concatenate([rt.get_ray_fan([fr * sep, 0, fr * lateral_shift], max_angle, nrays, wavelength) for fr in pupil_fractions], axis=0)

rays = ls.ray_trace(rays, rt.vacuum(), rt.vacuum())

# #######################################
# plot results
# #######################################
figh, ax = ls.plot(rays, colors=["k"] * nrays + ["b"] * nrays + ["r"] * nrays + ["g"] * nrays,
                   figsize=(16, 8))
ax.plot([ls.surfaces[12].center[2] - f_excitation, ls.surfaces[12].center[2] - f_excitation],
        [-radius, radius], 'k--', label="objective FP")
ax.legend()
figh.suptitle("ODT")
