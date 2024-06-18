"""
Numerically compute the point-spread function of a perfect imaging System,
i.e. one only limited by the objective lens NA
"""
import time
import numpy as np
from numpy import fft
from scipy.interpolate import griddata
import scipy.special
import raytrace.raytrace as rt
from raytrace.materials import Vacuum, Constant
# plotting
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import PowerNorm

plot_pupil = True
plot_rays = True

# physical data
# spatial units are mm
wavelength = 532e-6
k = 2*np.pi / wavelength
n1 = 1.4
na_obj = 1.35
# na_obj = 0.5
alpha_obj = np.arcsin(na_obj / n1)
mag = 100
f_tube_lens = 200
f1 = f_tube_lens / mag
r1 = na_obj * f1
na_img = na_obj / mag
alpha_img = np.arcsin(na_img)

fwhm_um = wavelength / na_obj / 2 / 1e-3
radial_zero_um = 0.61 * wavelength / na_obj / 1e-3
axial_zero_um = 2 * wavelength * n1 / na_obj ** 2

system = rt.System([rt.PerfectLens(f1, [0, 0, n1 * f1], [0, 0, 1], alpha_obj),  # O1
                    rt.FlatSurface([0, 0, n1 * f1 + f1], [0, 0, 1], 4 * r1),  # O1 pupil
                    rt.PerfectLens(f_tube_lens, [0, 0, n1 * f1 + f1 + f_tube_lens], [0, 0, 1], na_img),  # tube lens #1
                    rt.FlatSurface([0, 0, n1 * f1 + f1 + 2 * f_tube_lens], [0, 0, 1], r1)  # imaging plane
                    ],
                   [Vacuum(), Vacuum(), Vacuum()]
                   )

# define grid in pupil
# dxy = 10e-3
# nxy = int(2 * (6 * r1 // dxy) + 1)
dxy = 5e-3
nxy = int(2 * (3 * r1 // dxy) + 1)
xs_grid = dxy * np.arange(nxy)
xs_grid -= np.mean(xs_grid)
ys_grid = np.array(xs_grid, copy=True)
xx_pupil, yy_pupil = np.meshgrid(xs_grid, ys_grid)
npts = xx_pupil.size

extent_xy_pupil = [xx_pupil.min() - 0.5 * dxy, xx_pupil.max() + 0.5 * dxy,
                   yy_pupil.min() - 0.5 * dxy, yy_pupil.max() + 0.5 * dxy]

# output grid after fourier_xform
fxs = fft.fftshift(fft.fftfreq(nxy, dxy))
fys = fft.fftshift(fft.fftfreq(nxy, dxy))
xs_out = fxs * wavelength * f_tube_lens
ys_out = fys * wavelength * f_tube_lens
dxy_out = xs_out[1] - xs_out[0]

extent_xy_out = [xs_out.min() - 0.5 * dxy, xs_out.max() + 0.5 * dxy,
                 ys_out.min() - 0.5 * dxy, ys_out.max() + 0.5 * dxy]
extent_xy_out = [e / mag for e in extent_xy_out]

# launch rays and ray trace
nz = 51
dz = 0.0001
zs = dz * np.arange(nz, dtype=float)
zs -= np.mean(zs)

output_efield = np.zeros((nz, nxy, nxy), dtype=complex)
pupil_efield = np.zeros((nz, nxy, nxy), dtype=complex)
tstart = time.perf_counter()
for ii in range(nz):
    print("ray tracing z-plane %d/%d, elapsed time %0.2fs" % (ii + 1, nz, time.perf_counter() - tstart), end="\r")

    # ray tracing
    rays = rt.get_ray_fan([0, 0, zs[ii]], alpha_obj, 101, wavelength, nphis=51)
    # rays = rt.get_collimated_rays([dx, dy, dz], 1, 101, wavelength=wavelength, nphis=51,
    #                               normal=[0, np.sin(10*np.pi/180), np.cos(10*np.pi/180)])
    rays = system.ray_trace(rays, Constant(n1), Vacuum())

    # beam in pupil
    ind = 4
    xs = rays[ind, :, 0]
    ys = rays[ind, :, 1]
    phis = rays[ind, :, 6]

    to_use = np.logical_and(np.logical_not(np.isnan(xs)), np.logical_not(np.isnan(ys)))
    pts = np.stack((xs[to_use], ys[to_use]), axis=1)

    interp_pts = np.stack((xx_pupil.ravel(), yy_pupil.ravel()), axis=1)
    phis_interp = griddata(pts, phis[to_use], interp_pts).reshape(xx_pupil.shape)
    pupil_efield[ii] = np.exp(1j * phis_interp)
    pupil_efield[ii, np.sqrt(xx_pupil ** 2 + yy_pupil ** 2) > r1] = 0
    pupil_efield[ii, np.isnan(phis_interp)] = 0

    output_efield[ii] = fft.fftshift(fft.fft2(fft.ifftshift(pupil_efield[ii])))

    # plot rays in pupil
    if plot_pupil and np.abs(zs[ii]) < 1e-12:
        figh = plt.figure()
        plt.suptitle("dz = %0.3fmm" % zs[ii])

        ax = plt.subplot(2, 2, 1)
        ax.set_title("Rays and phases in pupil")
        # phis_plot = phis - np.nanmax(phis)
        im = ax.scatter(xs, ys, marker='.', c=phis, cmap="hsv", vmin=np.nanmin(phis), vmax=np.nanmax(phis))
        ax.add_artist(Circle((0, 0), radius=r1, color='k', fill=False))
        ax.axis("equal")
        plt.colorbar(im)
        ax.set_xlabel("$x$-position (mm)")
        ax.set_ylabel("$y$-position (mm)")
        ax.set_ylim([-r1, r1])
        ax.set_xlim([-r1, r1])

        ax = plt.subplot(2, 2, 2)
        ax.set_title("interpolated |E|")
        ax.imshow(np.abs(pupil_efield[ii]), extent=extent_xy_pupil, origin="lower")
        ax.add_artist(Circle((0, 0), radius=r1, color='k', fill=False))
        ax.set_xlabel("$x$-position (mm)")
        ax.set_ylabel("$y$-position (mm)")
        ax.set_ylim([-r1, r1])
        ax.set_xlim([-r1, r1])

        ax = plt.subplot(2, 2, 3)
        ax.set_title("interpolated arg(E)")
        ax.imshow(phis_interp, cmap="hsv", vmin=np.nanmin(phis), vmax=np.nanmax(phis),
                  extent=extent_xy_pupil, origin="lower")
        ax.add_artist(Circle((0, 0), radius=r1, color='k', fill=False))
        ax.set_xlabel("$x$-position (mm)")
        ax.set_ylabel("$y$-position (mm)")
        ax.set_ylim([-r1, r1])
        ax.set_xlim([-r1, r1])

        ax = plt.subplot(2, 2, 4)
        ax.set_title("ft")
        ax.imshow(np.abs(output_efield[ii]), extent=extent_xy_out, origin="lower")
        ax.set_xlim([-5e-3, 5e-3])
        ax.set_ylim([-5e-3, 5e-3])
        ax.set_xlabel("$x$-position (mm)")
        ax.set_ylabel("$y$-position (mm)")

    # ##################################
    # plot ray trace surfaces
    # ##################################
    if plot_rays and np.abs(zs[ii]) < 1e-12:
        system.plot_rays(rays)
        ax = plt.gca()
print("")

psf = np.abs(output_efield)**2
psf = psf / np.max(psf)

# ##################################
# plot PSF
# ##################################
extent_yz_out = [(ys_out.min() - 0.5 * dxy_out) / mag / 1e-3, (ys_out.max() + 0.5 * dxy_out) / mag / 1e-3,
                 (zs.min() - 0.5 * dz) / 1e-3, (zs.max() + 0.5 * dz) / 1e-3]

# compare with real PSF
r_out = np.sqrt(np.expand_dims(xs_out**2, axis=0) + np.expand_dims(ys_out**2, axis=1))
psf_xy_real = np.abs(scipy.special.j1(k*r_out / mag * na_obj) / (k*r_out / mag *na_obj))**2
psf_xy_real = psf_xy_real / psf_xy_real[nxy//2 + 1, nxy//2] * psf[nz // 2, nxy//2 + 1, nxy//2]

figh = plt.figure()
plt.suptitle("PSF")

ax = plt.subplot(2, 2, 1)
# ax.imshow(np.max(psf / np.max(psf), axis=2), cmap="bone", extent=extent_yz, origin="lower",
#           aspect="auto", interpolation="none", norm=PowerNorm(gamma=0.5))
ax.imshow(psf[:, :, nxy//2] / np.max(psf), cmap="bone", extent=extent_yz_out, origin="lower",
          aspect="auto", interpolation="none", norm=PowerNorm(gamma=0.5))
ax.set_xlim([-2.5, 2.5])
ax.set_xlabel("y (um)")
ax.set_ylabel("z (um)")

ax = plt.subplot(2, 2, 2)
ax.set_title("PSF, xy plane")
ax.imshow(psf[nz // 2] / np.max(psf), cmap="bone", extent=[e / 1e-3 for e in extent_xy_out],
          origin="lower", interpolation="none", norm=PowerNorm(gamma=0.5, vmin=0, vmax=1))
ax.set_xlim([-2.5, 2.5])
ax.set_ylim([-2.5, 2.5])
ax.set_xlabel("x (um)")
ax.set_ylabel("y (um)")
ax.add_artist(Circle((0, 0), radius=radial_zero_um, color='r', fill=False))

ax = plt.subplot(2, 2, 4)
ax.set_title("PSF theory, xy plane")
ax.imshow(psf_xy_real, cmap="bone", extent=[e / 1e-3 for e in extent_xy_out],
          origin="lower", interpolation="none", norm=PowerNorm(gamma=0.5, vmin=0, vmax=1))
ax.set_xlim([-2.5, 2.5])
ax.set_ylim([-2.5, 2.5])
ax.set_xlabel("x (um)")
ax.set_ylabel("y (um)")
ax.add_artist(Circle((0, 0), radius=radial_zero_um, color='r', fill=False))

ax = plt.subplot(2, 2, 3)
ax.set_title("|PSF - PSF theory|, xy plane")
ax.imshow(np.abs(psf[nz//2] - psf_xy_real), cmap="bone", extent=[e / 1e-3 for e in extent_xy_out],
          origin="lower", interpolation="none", norm=PowerNorm(gamma=0.1, vmin=0, vmax=1))
ax.set_xlim([-2.5, 2.5])
ax.set_ylim([-2.5, 2.5])
ax.set_xlabel("x (um)")
ax.set_ylabel("y (um)")
