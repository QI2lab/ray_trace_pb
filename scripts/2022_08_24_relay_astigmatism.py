"""
Test amount of astigmatism introduced by off-axis lens
"""
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import raytrace.raytrace as rt

wavelength = 0.785 # um
nrays = 19 # must be odd
offset = 5
beam_rad = 20e-3 * np.sqrt(1 + (3 / (np.pi * 20e-3**2 / (wavelength * 1e-3)))**2)

# give radii of curvature as if crown side facing left (i.e. as if collimated beam incident from left)

# AC508-100-B-ML
# N-LAK22/N-SF6HT
t100c = 13.0
r100c = 65.8
r100i = -56.
t100f = 2.0
r100f = -280.6
wd100 = 88.7
bfl100 = 91.5
efl100 = 100

# AC508-180-AB-ML
# N-LAK22/N-SF6
t180c = 9.5
r180c = 144.4
r180i = -115.4
t180f = 4.0
r180f = -328.2
wd180 = 170.6
bfl180 = 173.52
efl180 = 180

# AC508-300-AB-ML
# N-LAK22/N-SF6
t300c = 9.0
r300c = 167.7
r300i = -285.8
t300f = 4.0
r300f = np.inf
wd300 = 286.08
bfl300 = 289.81
efl300 = 300

radius = 25.4

z180 = 10
z100 = (t180c + t180f) + bfl180 + 101.5 #99.9609375
z300 = z100 + (t100c + t100f) + bfl100 + efl300
zend = z300 + (t300c + t300f) + bfl300
# zend = 998.579

surfaces = \
           [# AC508-180-AB-ML. Infinity focus to the left
            rt.spherical_surface(r180c, [offset, 0, z180 + np.abs(r180c)], radius),
            rt.spherical_surface(r180i, [offset, 0, z180 + t180c - np.abs(r180i)], radius),
            rt.spherical_surface(r180f, [offset, 0, z180 + t180c + t180f - np.abs(r180f)], radius),
            # rt.spherical_surface.get_on_axis(r180c, z180, radius),
            # rt.spherical_surface.get_on_axis(r180i, z180 + t180c, radius),
            # rt.spherical_surface.get_on_axis(r180f, z180 + t180c + t180f, radius),
            # AC508-100-B-ML. Infinity focus to the right
            rt.spherical_surface(-r100f, [offset, 0, z100 + np.abs(r100f)], radius),
            rt.spherical_surface(-r100i, [offset, 0, z100 + t100f + np.abs(r100i)], radius),
            rt.spherical_surface(-r100c, [offset, 0, z100 + t100f + t100c - np.abs(r100c)], radius),
            # AC508-300-AB-ML. Infinity focus to the left
            rt.spherical_surface.get_on_axis(r300c, z300, radius),
            rt.spherical_surface.get_on_axis(r300i, z300 + t300c, radius),
            rt.flat_surface([0, 0, z300 + t300c + t300f], [0, 0, 1], radius),
            # rt.spherical_surface.get_on_axis(r300f, z300 + t300c + t300f, radius),
            # final focal plane
            rt.flat_surface([0, 0, zend], [0, 0, 1], radius)
            ]

# surface materials
nlak22 = rt.nlak22()
nsf6 = rt.nsf6()
nsf6ht = rt.nsf6ht()

ns = [1,
      nlak22.n(wavelength), nsf6.n(wavelength), 1, #AC508-180-AB-ML
      nsf6ht.n(wavelength), nlak22.n(wavelength), 1, #AC508-100-B-ML
      nlak22.n(wavelength), nsf6.n(wavelength), 1, # AC508-300-AB-ML
      1]


rays = np.concatenate((rt.get_collimated_rays([0, 0, 0], beam_rad, nrays, wavelength), # distributed along meridional plane
                       rt.get_collimated_rays([0, 0, 0], beam_rad, nrays, wavelength, phi_start=np.pi/2), # distributed along sagittal plane
                       rt.get_collimated_rays([0, 0, 0], beam_rad, nrays, wavelength, nphis=100),
                       ), axis=0)

# do ray tracing
rays = rt.ray_trace_system(rays, surfaces, ns)

# estimate focus
f_meridional = rt.intersect_rays(rays[-2, nrays//2 - 1], rays[-2, nrays//2 + 1])
f_sagittal = rt.intersect_rays(rays[-2, nrays + nrays//2 - 1], rays[-2, nrays + nrays//2 + 1])

print(f"offset from lens center = {offset:.2f}mm")
print(f"meridional focus = {f_meridional[0, -1]:.2f}mm")
print(f"sagittal focus = {f_sagittal[0, -1]:.2f}mm")
print(f"meridional - sagittal focus = {f_meridional[0, -1] - f_sagittal[0, -1]:.5f}mm")

# estimate difference in radius of curvature
wo = 20e-3 * 180
zr = np.pi * wo**2 / (wavelength * 1e-3)
z_defocus = np.abs(f_meridional[0, -1] - f_sagittal[0, -1])
R = z_defocus * (1 + zr**2 / z_defocus**2)
print(f"for wo={wo:.3f}mm, R~{R / 1e3:.0f}m")


# plot results
figh, ax = rt.plot_rays(rays[:, :nrays], surfaces, colors="r", label="meridional", figsize=(16, 8))
rt.plot_rays(rays[:, nrays:2*nrays], surfaces, phi=np.pi/2, colors="b", label="sagittal", ax=ax)
figh.suptitle(f"ray trace, lens offset = {offset:.1f}mm")
ax.legend()
plt.show()

figh_spot, ax_spot = rt.plot_spot_diagram(rays[-1])
figh_spot.suptitle(f"spot_digram, lens offset = {offset:.1f}mm")