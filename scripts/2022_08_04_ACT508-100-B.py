"""
Evaluate Thorlabs achromat
"""
import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
import raytrace.raytrace as rt

lens_name = "AC508-100-B"
bfl_thor = 91.5
efl_thor = 100
# import materials
crown = rt.nlak22()
flint = rt.sf6()
# set lens parameters
radius = 25.4
lens_start = 300
tcrown = 13.0
rcrown = 65.8
rint = -56
tflint = 2.0
rflint = -280.6

# set ray numbers and wavelengths
nrays = 11
design_wavelengths = np.array([0.7065, 0.855, 1.015])
max_displacement = 10

# initialize arrays to store results
efls = np.zeros(len(design_wavelengths))
bfls = np.zeros(len(design_wavelengths))

# initialize figure showing spherical aberrations
figh_spherical = plt.figure()
figh_spherical.suptitle(f"Spherical aberration")
ax_spherical = figh_spherical.add_subplot(1, 1, 1)

# do ray tracing and paraxial calculations for all wavelengths
for ii, wl in enumerate(design_wavelengths):
    # define achromat oriented so collimated beam should enter from the left (i.e. the crown side)
    surfaces = [rt.spherical_surface.get_on_axis(rcrown, lens_start, radius),
                rt.spherical_surface.get_on_axis(rint, lens_start + tcrown, radius),
                rt.spherical_surface.get_on_axis(rflint, lens_start + tcrown + tflint, radius)
                ]

    # if there are n surfaces, want n+1 indices of refraction, giving values in between surfaces
    ns = [1, crown.n(wl), flint.n(wl), 1]

    # compute effective focal lengths and principle plans using paraxial optics
    abcd = rt.compute_paraxial(lens_start, surfaces, ns)
    dx, abcd_focal, eflx, dy, abcd_focal_y, efly = rt.find_paraxial_focus(abcd)

    # determine focal points
    f1, f2, pp1, pp2, efl1, efl2 = rt.find_cardinal_points(surfaces, ns, wl)
    efls[ii] = efl2
    ffl = lens_start - f1[0, 2]
    bfls[ii] = f2[0, 2] - (lens_start + tcrown + tflint)

    print(f"wavelength = {wl * 1e3:.0f}nm")
    print(f"efl (back) = {efls[ii]:.3f}mm")
    print(f"efl (front) = {efl1:.3f}mm")
    print(f"ffl = {ffl:.3f}mm")
    print(f"bfl = {bfls[ii]:.3f}mm")

    # add surface at focus
    # surfaces, ns = rt.auto_focus(surfaces, ns, mode="paraxial")
    surfaces, ns = rt.auto_focus(surfaces, ns, mode="collimated")

    # initialize collimated rays and ray trace to look at spherical aberration
    rays = rt.get_collimated_rays([0, 0, 0], max_displacement, nrays, wl)
    rays = rt.ray_trace_system(rays, surfaces, ns)

    # plot results
    figh , ax = rt.plot_rays(rays, surfaces, figsize=(16, 8))
    figh.suptitle(f"{lens_name:s} at {wl * 1e3:.0f}nm")
    ax.plot([f1[0, 2], f1[0, 2]], [-20, 20], 'r', label="FFP")
    ax.plot([f2[0, 2], f2[0, 2]], [-20, 20], 'g', label="BFP")
    ax.plot([pp1, pp1], [-20, 20], 'b', label="PP1")
    ax.plot([pp2, pp2], [-20, 20], 'm', label="PP2")
    ax.legend()

    # evaluate spherical aberration
    ray_heights = rays[0, nrays//2:, 0]
    foci = rt.intersect_rays([0, 0, 0, 0, 0, 1, 0, 2*np.pi/wl], rays[-1, nrays//2:])
    # add to spherical aberration plot
    ax_spherical.plot(ray_heights, foci[:, 2] - f2[0, 2], '-x', label=f"{wl * 1e3:.0f}nm")

ax_spherical.set_xlabel("Initial ray height (mm)")
ax_spherical.set_ylabel("Focus position (mm)")
ax_spherical.legend()

print(f"Mean bfl={np.mean(bfls):.3f}mm, expected {bfl_thor:.1f}mm")
print(f"Mean efl={np.mean(efls):.3f}mm, expected {efl_thor:.1f}mm")
