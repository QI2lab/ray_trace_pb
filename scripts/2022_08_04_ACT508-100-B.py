"""
Evaluate Thorlabs achromats
"""
import numpy as np
import matplotlib.pyplot as plt
import raytrace.raytrace as rt
from raytrace.materials import Nlak22, Bk7, Sf2, Nsf6, Nsf6ht, Sf10, Vacuum

# radius = 25.4 / 2
# bfl_thor = 143.68
# efl_thor = 150
# design_wavelengths = np.array([0.488, 0.707, 1.064])
# doublet = rt.Doublet(Nlak22(),
#                      Sf10(),
#                      radius_crown=87.9,
#                      radius_flint=np.inf,
#                      radius_interface=-105.6,
#                      thickness_crown=6.0,
#                      thickness_flint=3.0,
#                      aperture_radius=radius,
#                      input_collimated=True,
#                      names="AC254-150-AB"
#                      )

radius = 25.4
bfl_thor = 190.6
efl_thor = 200
design_wavelengths = np.array([0.4861, 0.5876, 0.6563])
doublet = rt.Doublet(Bk7(),
                     Sf2(),
                     radius_crown=106.2,
                     radius_flint=-409.4,
                     radius_interface=-92.1,
                     thickness_crown=10.6,
                     thickness_flint=6.0,
                     aperture_radius=radius,
                     input_collimated=True,
                     names="ACT508-200-A"
                     )

# radius = 25.4
# bfl_thor = 91.5
# efl_thor = 100
# design_wavelengths = np.array([0.7065, 0.855, 1.015])
# doublet = rt.Doublet(Nlak22(),
#                      Nsf6ht(),
#                      radius_crown=65.8,
#                      radius_flint=-280.6,
#                      radius_interface=-56,
#                      thickness_crown=13.0,
#                      thickness_flint=2.0,
#                      aperture_radius=radius,
#                      input_collimated=True,
#                      names="AC508-100-B"
#                      )

# radius = 25.4
# bfl_thor = 173.52
# efl_thor = 180
# design_wavelengths = np.array([0.488, 0.707, 1.064])
# doublet = rt.Doublet(Nlak22(),
#                      Nsf6(),
#                      radius_crown=144.4,
#                      radius_flint=-328.2,
#                      radius_interface=-115.4,
#                      thickness_crown=9.5,
#                      thickness_flint=4.0,
#                      aperture_radius=radius,
#                      input_collimated=True,
#                      names="AC508-180-AB"
#                      )

system = rt.System([rt.FlatSurface([0, 0, 0],
                                       [0, 0, 1],
                                       radius)],
                       [])
system = system.concatenate(doublet,
                            Vacuum(),
                            distance=10)


# set ray numbers and wavelengths
nrays = 101
max_displacement = 10

# initialize arrays to store results
efls = np.zeros(len(design_wavelengths))
bfls = np.zeros(len(design_wavelengths))
ffls = np.zeros(len(design_wavelengths))

# initialize figure showing spherical aberrations
figh_summary = plt.figure()
figh_summary.suptitle("Lens performance")
ax_spherical = figh_summary.add_subplot(1, 2, 1)
ax_spherical.set_title(f"Spherical aberration")
ax_spherical.axhline(0, c='k')

axf = figh_summary.add_subplot(1, 2, 2)
axf.set_title(f"focal shift")
axf.axhline(0, c='k')

# do ray tracing and paraxial calculations for all wavelengths
for ii, wl in enumerate(design_wavelengths):
    f1, f2, pp1, pp2, np1, np2, efl1, efl2 = system.get_cardinal_points(wl, Vacuum(), Vacuum())
    efls[ii] = efl2
    ffls[ii] = system.surfaces[1].paraxial_center[2] - f1[2]
    bfls[ii] = f2[2] - system.surfaces[3].paraxial_center[2]

    print(f"wavelength = {wl * 1e3:.0f}nm")
    print(f"efl (back) = {efls[ii]:.3f}mm")
    print(f"efl (front) = {efl1:.3f}mm")
    print(f"ffl = {ffls[ii]:.3f}mm")
    print(f"bfl = {bfls[ii]:.3f}mm")

    # add Surface at paraxial focus
    focused_system = system.concatenate(rt.System([rt.FlatSurface(f2,
                                                                  [0, 0, 1],
                                                                  radius)],
                                          []),
                                Vacuum(),
                                distance=None
                                )

    # initialize collimated rays and ray trace to look at spherical aberration
    rays = rt.get_collimated_rays([0, 0, 0], max_displacement, nrays, wl)
    rays = focused_system.ray_trace(rays,
                                    Vacuum(),
                                    Vacuum())

    # plot results
    figh, ax = focused_system.plot(rays, figsize=(16, 8))
    figh.suptitle(f"{doublet.names[0]:s} at {wl * 1e3:.0f}nm")
    # ax.axvline(f1[2], c='r', label="FFP")
    ax.axvline(f2[2], c='g', label="BFP")
    ax.axvline(pp1[2], c='b', label="PP1")
    ax.axvline(pp2[2], c='m', label="PP2")
    ax.legend()

    # evaluate spherical aberration
    ray_heights = rays[0, nrays//2:, 0]
    foci = rt.intersect_rays([0, 0, 0, 0, 0, 1, 0, 2*np.pi/wl], rays[-1, nrays//2:])
    # add to spherical aberration plot
    ax_spherical.plot(ray_heights,
                      foci[:, 2] - f2[2],
                      '-x',
                      label=f"{wl * 1e3:.0f}nm")

    # todo: compute MTF's

ax_spherical.set_xlabel("Initial ray height (mm)")
ax_spherical.set_ylabel("Focus position (mm)")
ax_spherical.legend()

axf.plot(design_wavelengths, bfls - bfls[1])
axf.set_xlabel("Wavelength (um)")
axf.set_ylabel("Focal shift (mm)")

print(f"Mean ffl={np.mean(ffls):.3f}mm")
print(f"Mean bfl={np.mean(bfls):.3f}mm, expected {bfl_thor:.1f}mm")
print(f"Mean efl={np.mean(efls):.3f}mm, expected {efl_thor:.1f}mm")

plt.show()
