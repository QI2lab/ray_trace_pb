"""
Ray trace "thin lens" and calculate Seidel aberration coefficients
"""
import numpy as np
import matplotlib.pyplot as plt
import ray_trace as rt

wavelength = 0.532

aperture_radius = 25.4
lens_start = 400
t200c = 10.6
t200f = 6
r200f = 409.4
r200i = 92.1
r200c = -106.2
bfl200 = 190.6
efl200 = 200
surfaces = [
    rt.flat_surface([0, 0, 0], [0, 0, 1], aperture_radius),
    # ACT508-200-A-ML
    rt.spherical_surface.get_on_axis(r200f, lens_start-0.1, aperture_radius, is_aperture_stop=True),
    rt.spherical_surface.get_on_axis(r200i, lens_start + t200f, aperture_radius),
    rt.spherical_surface.get_on_axis(r200c, lens_start + t200c + t200f, aperture_radius),
    #rt.flat_surface([0, 0, lens_start + t200c + t200f + 0], [0, 0, 1], aperture_radius)
    ]

bk7 = rt.bk7()
sf2 = rt.sf2()
ns = [1,
      1, sf2.n(wavelength), bk7.n(wavelength), 1,
      ]

# auto-focus
surfaces, ns = rt.auto_focus(surfaces, ns, mode="paraxial")

abcd = rt.compute_paraxial(400, surfaces, ns)

# ray trace
nrays = 101
rays = rt.get_ray_fan([0, 0, 0], 1*np.pi/180, nrays, wavelength)
rays = rt.ray_trace_system(rays, surfaces, ns)

# plot results
rt.plot_rays(rays, surfaces)

# test aberrations
abs = rt.compute_third_order_seidel(surfaces, ns)
for ii in range(abs.shape[0]):
    print(("surface %02d: " + 5 * "%+0.5e, ") % ((ii,) + tuple(abs[ii])))
print("total     : " + 5 * "%+0.5e, " % tuple(np.sum(abs, axis=0)))

# compare ray trace with ABCD
# uxs = rays[0, :, 3]
# uys = rays[0, :, 4]
# rays_in_abcd = np.vstack((rays[0, :, 0], uxs, rays[0, :, 1], uys, np.zeros(rays.shape[1])))
# rays_out_abcd = abcd.dot(rays_in_abcd)


# plt.figure()
# ax = plt.subplot(1, 1, 1)
# ax.plot(rays[-1, :, 0], rays[-1, :, 1], 'b.')
# ax.plot(rays_out_abcd[0, :], rays_out_abcd[2, :], 'rx')

