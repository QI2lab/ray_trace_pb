"""
Simulate full ODT optical setup, including extra coverglass to simulate flow cell

sample region consists of (1) layer of oil; (2) no. 1.5 coverslip (3) sample (water) (4) top coverglass (5) water immersion for detection objective
"""
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import raytrace.raytrace as rt
from pathlib import Path
import datetime

# #############################
# define phsyical parameters used in setup
# #############################

include_relay = False

wavelength = 0.785
dmd_tilt_angle = 20 * np.pi/180
aperture_radius = 25.4

# olympus 100x NA=1.3 oil-immersion objective (perfect lens)
f_excitation = 180 / 100
na_excitation = 1.3
fov_excitation = 0.130
alpha_excitation = np.arcsin(na_excitation / 1.5)

# olympus 60x NA=1.0 water-immersion objective (perfect lens) (formerly mitutoyo 50x NA 0.55)
f_detection = 180. / 60.
na_detection = 1.
alpha_detection = np.arcsin(na_detection / 1.333)

# sample region
n_coverglass = 1.5
thickness_coverslip = 0.17
n_oil = 1.5
# f = d1/n1 + d2/n2
thickness_oil = (f_excitation - thickness_coverslip / n_coverglass) * n_oil

n_sample = 1.333
thickness_sample = 0.1
n_top_coverslip = 1.5
# n_top_coverslip = 1.333
thickness_top_coverslip = 1.25 # measurement of Alexis' flow cell
# thickness_top_coverslip = 0.17
n_water = 1.333
# f = d1/n1 + d2/n2 + d3/n3
thickness_water_immersion = (f_detection - thickness_sample / n_sample - thickness_top_coverslip / n_top_coverslip) * n_water

# #############################
# define lenses
# #############################

l1 = rt.Doublet(names="ACT508-200-A-ML",
                material_crown=rt.Bk7(),
                material_flint=rt.Sf2(),
                radius_crown=106.2,
                radius_flint=-409.4,
                radius_interface=-92.1,
                thickness_crown=10.6,
                thickness_flint=6.,
                aperture_radius=aperture_radius,
                input_collimated=False)

l2 = rt.Doublet(names="AC508-100-A-ML",
                material_crown=rt.Nbaf10(),
                material_flint=rt.Sf10(),
                radius_crown=71.1,
                radius_flint=-363.1,
                radius_interface=-44.2,
                thickness_crown=16,
                thickness_flint=4,
                aperture_radius=aperture_radius,
                input_collimated=False
                )

l3 = rt.Doublet(names="AC508-400-A-ML",
                material_crown=rt.Bk7(),
                material_flint=rt.Sf2(),
                radius_crown=292.3,
                radius_flint=-398.5,
                radius_interface=-148.9,
                thickness_crown=8.,
                thickness_flint=8.,
                aperture_radius=aperture_radius,
                input_collimated=True
                )

l4 = rt.Doublet(names="AC508-300-A-ML",
                material_crown=rt.Bk7(),
                material_flint=rt.Sf2(),
                radius_crown=161.5,
                radius_flint=-580.8,
                radius_interface=-134,
                thickness_crown=6.0,
                thickness_flint=2.0,
                aperture_radius=aperture_radius,
                input_collimated=False
                )

obj1 = rt.System([rt.PerfectLens(f_excitation, [0, 0, 0], [0, 0, 1], alpha_excitation)],
                 [],
                 #names="obj1"
                 )

sample = rt.System([# oil
                    rt.FlatSurface([0, 0, 0], [0, 0, 1], aperture_radius),
                    # coverslip (assume focal plane right at coverslip)
                    rt.FlatSurface([0, 0, thickness_coverslip], [0, 0, 1], fov_excitation),
                    # water/sample
                    rt.FlatSurface([0, 0, thickness_coverslip + thickness_sample], [0, 0, 1], aperture_radius),
                    # top cover-glass
                    rt.FlatSurface([0, 0, thickness_coverslip + thickness_sample + thickness_top_coverslip], [0, 0, 1], aperture_radius)
                    ],
                    [rt.Constant(n_coverglass),  # coverslip
                     rt.Constant(n_sample),  # sample
                     rt.Constant(n_top_coverslip)],
                    names="sample")

obj2 = rt.System([rt.PerfectLens(f_detection, [0, 0, 0], [0, 0, 1], alpha_detection)],
                 [],
                 #names="obj2"
                 )

# l8 = rt.Doublet(name="AC508-200-A-ML",
#                 material_crown=rt.Bk7(),
#                 material_flint=rt.Sf2(),
#                 radius_crown=109.9,
#                 radius_flint=-376.3,
#                 radius_interface=-93.1,
#                 thickness_crown=8.5,
#                 thickness_flint=2.0,
#                 aperture_radius=aperture_radius,
#                 input_collimated=True
#                 )

l8 = rt.Doublet(names="AC508-180-AB-ML",
                material_crown=rt.Nlak22(),
                material_flint=rt.Nsf6(),
                radius_crown=144.4,
                radius_flint=-328.2,
                radius_interface=-115.4,
                thickness_crown=9.5,
                thickness_flint=4.0,
                aperture_radius=aperture_radius,
                input_collimated=True
                )

l9 = rt.Doublet(names="AC508-100-B-ML",
                material_crown=rt.Nlak22(),
                material_flint=rt.Nsf6ht(),
                radius_crown=65.8,
                radius_flint=-280.6,
                radius_interface=-56.0,
                thickness_crown=13.0,
                thickness_flint=2.0,
                aperture_radius=aperture_radius,
                input_collimated=False
                )

l10 = rt.Doublet(names="AC508-300-AB-ML",
                 material_crown=rt.Nlak22(),
                 material_flint=rt.Nsf6(),
                 radius_crown=167.7,
                 radius_flint=np.inf,
                 radius_interface=-285.8,
                 thickness_crown=9.0,
                 thickness_flint=4.0,
                 aperture_radius=aperture_radius,
                 input_collimated=True
                 )


# compute working distances and other paraxial info about lenses
fp1_a, fp1_b, _, _, efl1, _ = l1.get_cardinal_points(wavelength, rt.Vacuum(), rt.Vacuum())
wd1_left = (l1.surfaces[0].paraxial_center - fp1_a)[2]
wd1_right = (fp1_b - l1.surfaces[-1].paraxial_center)[2]

fp2_a, fp2_b, _, _, efl2, _ = l2.get_cardinal_points(wavelength, rt.Vacuum(), rt.Vacuum())
wd2_left = (l2.surfaces[0].paraxial_center - fp2_a)[2]
wd2_right = (fp2_b - l2.surfaces[-1].paraxial_center)[2]

fp3_a, fp3_b, _, _, efl3, _ = l3.get_cardinal_points(wavelength, rt.Vacuum(), rt.Vacuum())
wd3_left = (l3.surfaces[0].paraxial_center - fp3_a)[2]
wd3_right = (fp3_b - l3.surfaces[-1].paraxial_center)[2]

fp4_a, fp4_b, _, _, efl4, _ = l4.get_cardinal_points(wavelength, rt.Vacuum(), rt.Vacuum())
wd4_left = (l4.surfaces[0].paraxial_center - fp4_a)[2]
wd4_right = (fp4_b - l4.surfaces[-1].paraxial_center)[2]

fp5_a, fp5_b, _, _, efl5, _ = obj1.get_cardinal_points(wavelength, rt.Vacuum(), rt.Vacuum())
wd5_left = (obj1.surfaces[0].paraxial_center - fp5_a)[2]
# wd5_right = (fp5_b - obj1.surfaces[-1].paraxial_center)[2]

fp7_a, fp7_b, _, _, efl7, _ = obj2.get_cardinal_points(wavelength, rt.Vacuum(), rt.Vacuum())
# wd7_left = (obj2.surfaces[0].paraxial_center - fp7_a)[2]
wd7_right = (fp7_b - obj2.surfaces[-1].paraxial_center)[2]

fp8_a, fp8_b, _, _, efl8, _ = l8.get_cardinal_points(wavelength, rt.Vacuum(), rt.Vacuum())
wd8_left = (l8.surfaces[0].paraxial_center - fp8_a)[2]
wd8_right = (fp8_b - l8.surfaces[-1].paraxial_center)[2]

fp9_a, fp9_b, _, _, efl9, _ = l9.get_cardinal_points(wavelength, rt.Vacuum(), rt.Vacuum())
wd9_left = (l9.surfaces[0].paraxial_center - fp9_a)[2]
wd9_right = (fp9_b - l9.surfaces[-1].paraxial_center)[2]

fp10_a, fp10_b, _, _, efl10, _ = l10.get_cardinal_points(wavelength, rt.Vacuum(), rt.Vacuum())
wd10_left = (l10.surfaces[0].paraxial_center - fp10_a)[2]
wd10_right = (fp10_b - l10.surfaces[-1].paraxial_center)[2]

# create optical System using paraxial working distances to set spacing between lenses
ls = l1.concatenate(l2, rt.Vacuum(), wd1_right + wd2_left)
ls = ls.concatenate(l3, rt.Vacuum(), wd2_right + wd3_left)
ls = ls.concatenate(l4, rt.Vacuum(), wd3_right + wd4_left)
ls = ls.concatenate(obj1, rt.Vacuum(), wd4_right + wd5_left)
ls = ls.concatenate(sample, rt.Constant(n_oil), thickness_oil)
ls = ls.concatenate(obj2, rt.Constant(n_water), thickness_water_immersion) # detection objective
ls = ls.concatenate(l8, rt.Vacuum(), wd7_right + wd8_left) # tube lens

if include_relay:
    ls = ls.concatenate(l9, rt.Vacuum(), wd8_right + wd9_left) # relay lens #1
    ls = ls.concatenate(l10, rt.Vacuum(), wd9_right + wd10_left) # relay lens #2
    ls = ls.concatenate(rt.System([rt.FlatSurface([0, 0, 0], [0, 0, 1], aperture_radius)], []),  # add camera
                        rt.Vacuum(), wd10_right)
else:
    ls = ls.concatenate(rt.System([rt.FlatSurface([0, 0, 0], [0, 0, 1], aperture_radius)], []),  # add camera
                        rt.Vacuum(), wd8_right)

# #######################################
# ray tracing
# #######################################
max_angle = 0.5 * np.pi/180
# edge of detection pupil
sep = f_detection * na_detection * (f_excitation/f_detection * efl3/efl4 * efl1/efl2)
# account for tilted DMD
lateral_shift = -np.sin(dmd_tilt_angle) * sep

# generate rays
pupil_fractions = [0, 1/3, 2/3, 0.95]
nrays = 21
rays = np.concatenate([rt.get_ray_fan([fr * sep, 0, fr * lateral_shift - wd1_left], max_angle, nrays, wavelength) for fr in pupil_fractions], axis=0)

# ray trace
rays = ls.ray_trace(rays, rt.Vacuum(), rt.Vacuum())

# #######################################
# plot results
# #######################################
figh, ax = ls.plot(rays, colors=["k"] * nrays + ["b"] * nrays + ["r"] * nrays + ["g"] * nrays,
                   figsize=(16, 8))
ax.plot([ls.surfaces[12].center[2] - f_excitation, ls.surfaces[12].center[2] - f_excitation],
        [-aperture_radius, aperture_radius], 'm--', label="objective FP")
ax.legend()
figh.suptitle("ODT optical System", fontsize=16)


# zoomed in version for saving
saving = False
if saving:
    figh_save, ax = ls.plot(rays,
                            colors=["k"] * nrays + ["b"] * nrays + ["r"] * nrays + ["g"] * nrays,
                            show_names=False,
                            figsize=(16, 8))
    ax.set_xlim([1800, 2230])
    ax.set_ylim([-15, 15])
    tstamp = datetime.datetime.now().strftime('%Y_%m_%d_%H;%M;%S')
    figh_save.savefig(Path.home() / "Desktop" / f"{tstamp:s}_optical_system.pdf", bbox_inches="tight")
