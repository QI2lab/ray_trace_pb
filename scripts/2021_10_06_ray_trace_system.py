"""
Simulate full SIM/ODT optical setup
"""
import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
import raytrace.raytrace as rt

wavelength_align = 0.532 # todo: simulate alignment
wavelength_odt = 0.785
wavelengths_sim = [0.465, 0.532, 0.635]
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
# perfect lens

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

focal_plane = l5e_odt + 1.5 * 1.8

# water_end = focal_plane + 1
water_thickness = 1

# water_focal_shift = 1.5 * 0.275 - 0.017 - 0.01
# water_focal_shift = 0.3 - 0.0399 + (0.15 - 0.023)
# water_focal_shift = 0.1375 - 0.0042
water_focal_shift = 0.1275 * 2
l6s_odt = l5e_odt + 1.5 * 1.8 + 4 + water_focal_shift # for 1.5mm water
# l6s_odt = l5e_odt + 1.5 * 1.8 + 4 + 0.275 # for 1mm water
# l6s_odt = l5e_odt + 1.5 * 1.8 + 4 + 0.5 * 0.275 # for 0.5mm water
# l6s_odt = l5e_odt + 1.5 * 1.8 + 4 + 0.1 * 0.275 # for 0.1mm water
l6e_odt = l6s_odt

# keep position fixed regardless of water focal shift
l7s_odt = l6s_odt + (4 + efl200_old - water_focal_shift)
l7s_odt = l7s_odt + t200c + t200f

camera_pos = l7s_odt + bfl200_old + 11.2

surfaces_excitation_odt =  \
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
                # olympus 100x NA 1.3 oil immersion objective
                rt.perfect_lens(1.8, [0, 0, l5s_odt], [0, 0, 1], 1.8 * 1.3),
                # focal plane: assume space between is oil + coverslip both with n=1.5
                rt.flat_surface([0, 0, focal_plane], [0, 0, 1], 0.130)]

surfaces_detection_odt = \
               [# space between focus and edge of water
                rt.flat_surface([0, 0, focal_plane + water_thickness], [0, 0, 1], 1),
                # mitutoyo 50x NA 0.55
                rt.perfect_lens(4, [0, 0, l6s_odt], [0, 0, 1], 4 * 0.55),
                # ACT08-200-A-ML
                rt.spherical_surface.get_on_axis(-r200c_old, l7s_odt, radius),
                rt.spherical_surface.get_on_axis(-r200i_old, l7s_odt + t200c_old, radius),
                rt.spherical_surface.get_on_axis(-r200f_old, l7s_odt + t200c_old + t200f_old, radius),
                # final focal plane
                rt.flat_surface([0, 0, camera_pos], [0, 0, 1], 25.4)
                ]
surfaces_odt = surfaces_excitation_odt + surfaces_detection_odt

ns_excitation_odt = [1,
                    sf2.n(wavelength_odt), bk7.n(wavelength_odt), 1,
                    sf10.n(wavelength_odt), nbaf10.n(wavelength_odt), 1,
                    bk7.n(wavelength_odt), sf2.n(wavelength_odt), 1,
                    sf2.n(wavelength_odt), bk7.n(wavelength_odt), 1,
                    1.5]
ns_detection_odt = [1.33,
                    1, 1,
                    bk7.n(wavelength_odt), sf2.n(wavelength_odt), 1,
                    1]

ns_odt = ns_excitation_odt + ns_detection_odt

# determine focal points
f1, f2, pp1, pp2, efl1, efl2 = rt.find_cardinal_points(surfaces_odt, ns_odt, wavelength_odt)
ffl = l1s_odt - f1[0, 2]
bfl = f2[0, 2] - l5e_odt
print("efl (back) = %0.3fmm" % efl1)
print("efl (front) = %0.3fmm" % (pp1 - f1[0, 2]))
print("ffl = %0.3fmm" % ffl)
print("bfl = %0.3fmm" % bfl)

# autofocus
# surfaces_odt, ns_odt = rt.auto_focus(surfaces_odt, ns_odt, wavelength)

# dm = 7.56
# mirror_rad = 5
# sigma = mirror_rad # treating mirror_rad as ~sigma
# sigma_frq = 1 / (2*np.pi * sigma)
# max_angle = np.arcsin(wavelength * sigma_frq)
max_angle = 0.5 * np.pi/180

sep = 4 * 0.55 * (1.8/4 * 400/300 * 200/100) # edge of mitutoyo pupil
# sep = 1.8 * 1.3 * (400/300 * 200/100) # edge of olympus pupil
lateral_shift = 0 * -np.sin(20 * np.pi/180) * sep
nrays = 7
rays = np.concatenate((rt.get_ray_fan([0, 0, 0], max_angle, nrays, wavelength_odt),
                       rt.get_ray_fan([1/3 * sep, 0, 1/3 * lateral_shift], max_angle, nrays, wavelength_odt),
                       rt.get_ray_fan([2/3 * sep, 0, 2/3 * lateral_shift], max_angle, nrays, wavelength_odt),
                       rt.get_ray_fan([sep, 0, lateral_shift], max_angle, nrays, wavelength_odt)),
                      axis=0)


rays = rt.ray_trace_system(rays, surfaces_odt, ns_odt)

# plot results
figh = rt.plot_rays(rays, surfaces_odt, colors=["k"] * nrays + ["b"] * nrays + ["r"] * nrays + ["g"] * nrays, figsize=(16, 8))
ax = plt.gca()
ax.plot([l5s_odt - 1.8, l5s_odt - 1.8], [-10, 10], 'k--')
plt.suptitle("ODT")

# fluorescence odt
rays_fl_odt = rt.get_ray_fan([0, 0, focal_plane], np.arcsin(0.55 / 1.33), nrays, wavelength_odt)
rays_fl_odt = rt.ray_trace_system(rays_fl_odt, surfaces_detection_odt, ns_detection_odt)

figh = rt.plot_rays(rays_fl_odt, surfaces_detection_odt, figsize=(16, 8))
ax = plt.gca()
plt.suptitle("ODT, fluorescence detection")

# #############################
# Imaging system for SIM = DMD in imaging plane
# #############################
l1s_sim = d_dmd_lens
l1e_sim = l1s_sim + t200c + t200f

l2s_sim = l1e_sim + d_200_400
l2e_sim = l2s_sim + t400c + t400f

l3s_sim = l2e_sim + d_400_300
l3e_sim = l3s_sim + t300c + t300f

l4s_sim = l3e_sim + d_300_obj
l4e_sim = l4s_sim

surfaces_sim = [# ACT508-200-A-ML
                rt.spherical_surface.get_on_axis(r200f, l1s_sim, radius),
                rt.spherical_surface.get_on_axis(r200i, l1s_sim + t200f, radius),
                rt.spherical_surface.get_on_axis(r200c, l1s_sim + t200c + t200f, radius),
                # AC508-400-A-ML
                rt.spherical_surface.get_on_axis(-r400c, l2s_sim, radius),
                rt.spherical_surface.get_on_axis(-r400i, l2s_sim + t400c, radius),
                rt.spherical_surface.get_on_axis(-r400f, l2s_sim + t400c + t400f, radius),
                # AC508-300-A-ML
                rt.spherical_surface.get_on_axis(r300f, l3s_sim, radius),
                rt.spherical_surface.get_on_axis(r300i, l3s_sim + t300f, radius),
                rt.spherical_surface.get_on_axis(r300c, l3s_sim + t300c + t300f, radius),
                # objective
                rt.perfect_lens(1.8, [0, 0, l4s_sim], [0, 0, 1], 1.8 * 1.3),
                rt.flat_surface([0, 0, l4s_sim + 1.5 * 1.8], [0, 0, 1], 0.13)
                ]

ns_sim_colors = [[1,
                 sf2.n(wavelength), bk7.n(wavelength), 1,
                 bk7.n(wavelength), sf2.n(wavelength), 1,
                 sf2.n(wavelength), bk7.n(wavelength), 1,
                 1.5, 1.5] for wavelength in wavelengths_sim]

# surfaces_sim, ns_sim = rt.auto_focus(surfaces_sim, ns_sim, wavelength)

max_angle = 0.89 * np.pi/180
sep = 10
lateral_shift = 0 * -np.sin(20 * np.pi/180) * sep
nrays = 25
rays_sim = [np.concatenate((rt.get_ray_fan([0, 0, 0], max_angle, nrays, wavelength),
                           rt.get_ray_fan([0.1*sep, 0, 0.5*lateral_shift], max_angle, nrays, wavelength),
                           rt.get_ray_fan([0.5*sep, 0, 0.5*lateral_shift], max_angle, nrays, wavelength),
                           rt.get_ray_fan([sep, 0, lateral_shift], max_angle, nrays, wavelength)),
                           axis=0) for wavelength in wavelengths_sim]


for ii in range(len(rays_sim)):
    rays_sim[ii] = rt.ray_trace_system(rays_sim[ii], surfaces_sim, ns_sim_colors[ii])


# plot results
rays_sim_all = np.concatenate(rays_sim, axis=1)
figh = rt.plot_rays(rays_sim_all, surfaces_sim, colors=["b"] * (4 * nrays) + ["g"] * (4 * nrays) + ["r"] * (4 * nrays),
                    figsize=(16, 8))
ax = plt.gca()
plt.suptitle("SIM")

# phase at first pupil
pupil1 = rt.flat_surface([0, 0, l1e_sim + 200], [0, 0, 1], radius)
rays_pupil1 = rt.ray_trace_system(rays_sim[0][6], [pupil1], [1, 1])

figh = plt.figure()
for ii in range(4):
    plt.plot(rays_pupil1[-1, nrays*ii:nrays*(ii+1), 0],
             rays_pupil1[-1, nrays*ii:nrays*(ii+1), 6] - np.min(rays_pupil1[-1, nrays*ii:nrays*(ii+1), 6]))
ax = plt.gca()
ax.set_xlabel("x-position at first pupil (mm)")
ax.set_ylabel("phase")
ax.set_title("phase versus 1st pupil position for rays originating at different heights")

pupil_pos_on_axis = rays_pupil1[-1, :nrays, 0]
phases_on_axis = rays_pupil1[-1, :nrays, 6] - np.min(rays_pupil1[-1, :nrays, 6])
pfit = np.polyfit(pupil_pos_on_axis, phases_on_axis, 4)
spherical = pfit[0] / (6 * np.sqrt(5))
defocus = (pfit[2] - 6 * np.sqrt(5) * spherical) / (2 * np.sqrt(3))
piston = pfit[-1] - defocus + spherical

# phase at last pupil
pupil_last = rt.flat_surface([0, 0, l4s_sim - 1.8], [0, 0, 1], radius)
rays_pupil_last = rt.ray_trace_system(rays_sim[0][18], [pupil_last], [1, 1])

figh = plt.figure()
for ii in range(4):
    plt.plot(rays_pupil_last[-1, nrays*ii:nrays*(ii+1), 0],
             rays_pupil_last[-1, nrays*ii:nrays*(ii+1), 6] - np.min(rays_pupil_last[-1, nrays*ii:nrays*(ii+1), 6]))
ax = plt.gca()
ax.set_xlabel("x-position at pupil (mm)")
ax.set_ylabel("phase")
ax.set_title("phase versus last pupil position for rays originating at different points")
