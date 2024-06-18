"""
Ray trace "thin lens" and calculate Seidel aberration coefficients
"""
import numpy as np
import raytrace.raytrace as rt
from raytrace.materials import Vacuum, Sf2, Bk7

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

system = rt.System([rt.FlatSurface([0, 0, 0], [0, 0, 1], aperture_radius),
                    # ACT508-200-A-ML
                    rt.SphericalSurface.get_on_axis(r200f, lens_start - 0.1, aperture_radius, is_aperture_stop=True),
                    rt.SphericalSurface.get_on_axis(r200i, lens_start + t200f, aperture_radius),
                    rt.SphericalSurface.get_on_axis(r200c, lens_start + t200c + t200f, aperture_radius),
                    #rt.FlatSurface([0, 0, lens_start + t200c + t200f + 0], [0, 0, 1], aperture_radius)
                    ],
                   [Vacuum(), Sf2(), Bk7()]
                   )

# auto-focus
f = system.auto_focus(wavelength, mode="paraxial-focused")
abcd = system.get_ray_transfer_matrix(wavelength, Vacuum(), Vacuum())

# ray trace
nrays = 101
rays = rt.get_ray_fan([0, 0, 0], 1*np.pi/180, nrays, wavelength)
rays = system.ray_trace(rays, Vacuum(), Vacuum())
system.plot(rays)

# test aberrations
abs = system.compute_third_order_seidel(wavelength)
for ii in range(abs.shape[0]):
    print(("Surface %02d: " + 5 * "%+0.5e, ") % ((ii,) + tuple(abs[ii])))
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

