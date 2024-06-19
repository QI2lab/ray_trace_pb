"""
Calculate ideal PSF of "snouty" oblique plane microscope (OPM)
"""
import time
from pathlib import Path
import numpy as np
from numpy import fft
#import tifffile
import matplotlib.pyplot as plt
import raytrace.raytrace as rt
from raytrace.materials import Vacuum, Constant

plot_results = True

# parameters
wavelength = 532e-6
aperture_rad = 2

# O1
n1 = 1.4
na1 = 1.35
alpha1 = np.arcsin(na1 / n1)
mag1 = 100
f1 = 200 / mag1
r1 = na1 * f1
f_tube_lens_1 = 200

# O2
n2 = 1
na2 = 0.95
alpha2 = np.arcsin(na2 / n2)
mag2 = 40
f2 = 200 / mag2
r2 = na2 * f2
#f_tube_lens_2 = 357
# modified to theoretically correct length...I could see the effect of even a few percent mag error of using 357mm
f_tube_lens_2 = f_tube_lens_1 / f1 *  f2 / n1
remote_mag = f_tube_lens_1 / f1 * f2 / f_tube_lens_2 # should = n1 / n2 = 1.4

# O3
theta = 30 * np.pi/180
n3 = 1.51
na3 = 1
alpha3 = np.arcsin(na3 / n3)
mag3 = 100
f3 = 200 / mag3
r3 = na3 * f3
o3_normal = np.array([-np.sin(theta), 0, np.cos(theta)])
f_tube_lens_3 = 200

# total magnification
total_mag = remote_mag * mag3

# positions
p_o1 = n1 * f1 # O1 position
p_pupil_o1 = p_o1 + f1 # O1 pupil position
p_t1 = p_o1 + f1 + f_tube_lens_1 # tube lens 1 position
p_t2 = p_t1 + f_tube_lens_1 + f_tube_lens_2 # tube lens 2 position
p_pupil_o2 = p_t2 + f_tube_lens_2 # O2 pupil position
p_o2 = p_t2 + f_tube_lens_2 + f2 # O2 position
p_remote_focus = p_o2 + n2 * f2
p_o3 = np.array([0, 0, p_remote_focus]) + n3 * f3 * o3_normal
p_pupil_o3 = p_o3 + f3 * o3_normal
p_t3 = p_o3 + (f3 + f_tube_lens_3) * o3_normal
p_imag = p_t3 + f_tube_lens_3 * o3_normal

system = rt.System([rt.PerfectLens(f1, [0, 0, p_o1], [0, 0, 1], alpha1),  # O1
                    rt.FlatSurface([0, 0, p_pupil_o1], [0, 0, 1], n1 * f1),  # O1 pupil
                    rt.PerfectLens(f_tube_lens_1, [0, 0, p_t1], [0, 0, 1], alpha1),  # tube lens #1
                    rt.PerfectLens(f_tube_lens_2, [0, 0, p_t2], [0, 0, 1], alpha2),  # tube lens #2
                    rt.FlatSurface([0, 0, p_pupil_o2], [0, 0, 1], n2 * f2),  # pupil of O2
                    rt.PerfectLens(f2, [0, 0, p_o2], [0, 0, 1], alpha2),  # O2
                    rt.FlatSurface([0, 0, p_remote_focus], o3_normal, r2),  # snouty nose cone
                    rt.PerfectLens(f3, p_o3, o3_normal, alpha3),  # O3
                    rt.FlatSurface(p_pupil_o3, o3_normal, r3),  # pupil of 03
                    rt.PerfectLens(f_tube_lens_3, p_t3, o3_normal, alpha3),  # tube lens #3
                    rt.FlatSurface(p_imag, o3_normal, aperture_rad)],
                   [Vacuum(),
                    Vacuum(),
                    Vacuum(),
                    Vacuum(),
                    Vacuum(),
                    Constant(n2),
                    Constant(n3),
                    Vacuum(),
                    Vacuum(),
                    Vacuum()]
                   )
# materials = [rt.Constant(n1), , rt.Vacuum()]

# setup grid in 03 pupil
dxy = 5e-3
nxy = int(2 * (3 * r3 // dxy) + 1)
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
xs_out = fxs * wavelength * f_tube_lens_3
ys_out = fys * wavelength * f_tube_lens_3
dxy_out = xs_out[1] - xs_out[0]

extent_xy_out = [xs_out.min() - 0.5 * dxy, xs_out.max() + 0.5 * dxy,
                 ys_out.min() - 0.5 * dxy, ys_out.max() + 0.5 * dxy]
extent_xy_out = [e / total_mag for e in extent_xy_out]


# launch rays and ray trace
tstart = time.perf_counter()

npos = 201
dx_pos = dxy_out / total_mag * np.cos(theta)
xs = dx_pos * np.arange(npos, dtype=float)
xs -= np.mean(xs)
# xs = np.array([0.001])

output_efield = np.zeros((npos, nxy, nxy), dtype=complex)
pupil_efield = np.zeros((npos, nxy, nxy), dtype=complex)
for ii in range(npos):
    print(f"ray tracing z-plane {ii+1:d}/{npos:d}, "
          f"elapsed time {time.perf_counter() - tstart:.2f}s", end="\r")

    # zpos = xs[ii] * np.tan(theta)
    zpos = 0
    rays = rt.get_ray_fan([xs[ii], 0, zpos], alpha1, 101, wavelength=wavelength, nphis=51)
    rays = system.ray_trace(rays, Constant(n1), Vacuum())

    # ##################################
    # plot rays in O1 pupil
    # ##################################
    rays_pupil_o1 = rays[4]
    # rays_pupil_o1 = rays[2]
    x_o1 = rays_pupil_o1[:, 0]
    y_o1 = rays_pupil_o1[:, 1]
    phi_o1 = rays_pupil_o1[:, -2]

    # ##################################
    # plot rays in O2 pupil
    # ##################################
    rays_pupil_o2 = rays[10]
    x_o2 = rays_pupil_o2[:, 0]
    y_o2 = rays_pupil_o2[:, 1]
    phi_o2 = rays_pupil_o2[:, -2]

    # ##################################
    # plot rays in 03 pupil
    # ##################################
    rays_pupil_o3 = rays[-5]

    # basis vectors
    na = np.array([np.cos(theta), 0, np.sin(theta)])
    nc = surfaces[-1].normal

    nb = np.cross(nc, na)
    nb = nb / np.linalg.norm(nb)
    # Surface center
    c = surfaces[-3].center

    # x-like vectors
    x_o3 = np.sum((rays_pupil_o3[:, :3] - np.expand_dims(c, axis=0)) * np.expand_dims(na, axis=0), axis=1)
    y_o3 = np.sum((rays_pupil_o3[:, :3] - np.expand_dims(c, axis=0)) * np.expand_dims(nb, axis=0), axis=1)
    phi_o3 = rays_pupil_o3[:, -2]

    # ##################################
    # interpolate electric field in 03 pupil
    # ##################################
    to_use = np.logical_and(np.logical_not(np.isnan(x_o3)), np.logical_not(np.isnan(y_o3)))
    pts = np.stack((x_o3[to_use], y_o3[to_use]), axis=1)

    interp_pts = np.stack((xx_pupil.ravel(), yy_pupil.ravel()), axis=1)
    phis_interp = griddata(pts, phi_o3[to_use], interp_pts).reshape(xx_pupil.shape)
    pupil_efield[ii] = np.exp(1j * phis_interp)
    pupil_efield[ii, np.sqrt(xx_pupil ** 2 + yy_pupil ** 2) > r1] = 0
    pupil_efield[ii, np.isnan(phis_interp)] = 0

    output_efield[ii] = fft.fftshift(fft.fft2(fft.ifftshift(pupil_efield[ii])))

    # ##################################
    # plot results
    # ##################################
    if np.mod(ii, npos // 4) == 0 and plot_results:
        # ##################################
        # ray trace
        # ##################################
        system.plot(rays)
        ax = plt.gca()
        ax.axis("equal")
        ax.set_title(f"ray trace, input position = (x, y, z) = ({xs[ii]:.5f}, 0, 0)")

        # draw imaging plane
        l = 1
        ax.plot([-np.sin(theta) * l, np.sin(theta) * l], [-np.cos(theta) * l, np.cos(theta) * l], 'k')

        # ##################################
        # phases in pupil
        # ##################################
        figh = plt.figure()
        plt.suptitle(f"input position = (x, y, z) = ({xs[ii]:.5f}, 0, 0)")
        grid = plt.GridSpec(2, 3)

        # O1 pupil
        ax = plt.subplot(grid[0, 0])
        ax.set_title("Rays and phases in O1 pupil")

        im = ax.scatter(x_o1,
                        y_o1,
                        marker='.',
                        c=phi_o1,
                        cmap="hsv",
                        vmin=np.nanmin(phi_o1),
                        vmax=np.nanmax(phi_o1))
        ax.add_artist(Circle((0, 0), radius=n1*f1, color='k', fill=False))
        ax.axis("equal")
        plt.colorbar(im)
        ax.set_xlabel("$x$-position (mm)")
        ax.set_ylabel("$y$-position (mm)")
        ax.set_ylim([-na1*f1, na1*f1])
        ax.set_xlim([-na1*f1, na1*f1])

        # O2 pupil
        ax = plt.subplot(grid[0, 1])
        ax.set_title("Rays and phases in O2 pupil")

        im = ax.scatter(x_o2, y_o2, marker='.', c=phi_o2, cmap="hsv", vmin=np.nanmin(phi_o2), vmax=np.nanmax(phi_o2))
        ax.add_artist(Circle((0, 0), radius=n2*f2, color='k', fill=False))
        ax.axis("equal")
        plt.colorbar(im)
        ax.set_xlabel("$x$-position (mm)")
        ax.set_ylabel("$y$-position (mm)")
        ax.set_ylim([-na2*f2, na2*f2])
        ax.set_xlim([-na2*f2, na2*f2])

        # O3 pupil
        ax = plt.subplot(grid[0, 2])
        ax.set_title("Rays and phases in O3 pupil")

        im = ax.scatter(x_o3, y_o3, marker='.', c=phi_o3, cmap="hsv", vmin=np.nanmin(phi_o3), vmax=np.nanmax(phi_o3))
        ax.add_artist(Circle((0, 0), radius=na3*f3, color='k', fill=False))
        ax.axis("equal")
        plt.colorbar(im)
        ax.set_xlabel("position along $n_a$ (mm)")
        ax.set_ylabel("position along $n_b$ (mm)")
        ax.set_ylim([-r3, r3])
        ax.set_xlim([-r3, r3])

        # interpolated O3
        ax = plt.subplot(grid[1, 0])
        ax.set_title("interpolated |E| O3")

        im = ax.imshow(np.abs(pupil_efield[ii]), extent=extent_xy_pupil, origin="lower")
        ax.add_artist(Circle((0, 0), radius=r3, color='k', fill=False))
        ax.set_xlabel("position along $n_a$ (mm)")
        ax.set_ylabel("position along $n_b$ (mm)")
        ax.set_ylim([-r3, r3])
        ax.set_xlim([-r3, r3])

        ax = plt.subplot(grid[1, 1])
        ax.set_title("interpolated arg(E) O3")
        ax.imshow(phis_interp,
                  cmap="hsv",
                  vmin=np.nanmin(phi_o3),
                  vmax=np.nanmax(phi_o3),
                  extent=extent_xy_pupil,
                  origin="lower")
        ax.add_artist(Circle((0, 0), radius=r3, color='k', fill=False))
        ax.set_xlabel("position along $n_a$ (mm)")
        ax.set_ylabel("position along $n_b$ (mm)")
        ax.set_ylim([-r3, r3])
        ax.set_xlim([-r3, r3])

        # output field
        ax = plt.subplot(grid[1, 2])
        ax.set_title("PSF")

        extent_xy_out_um = [e / 1e-3 for e in extent_xy_out]
        ax.imshow(np.abs(output_efield[ii])**2, extent=extent_xy_out_um, origin="lower", cmap="bone")
        ax.set_xlim([-2.5, 2.5])
        ax.set_ylim([-2.5, 2.5])
        ax.set_xlabel("$x$-position (um)")
        ax.set_ylabel("$y$-position (um)")
print("")

for ii in range(0, npos, 20):
    figh = plt.figure()
    plt.suptitle("imaged %d, x = %0.3fum" % (ii, xs[ii] / 1e-3))
    ax = plt.subplot(1, 1, 1)
    ax.imshow(np.abs(output_efield[ii])**2, extent=extent_xy_out_um, origin="lower", cmap="bone")
    ax.set_xlim([-4, 4])
    ax.set_ylim([-2, 2])
    ax.set_xlabel("na position (um)")
    ax.set_ylabel("nb position (um)")


# ##################################
# convert psf to coverslip frame
# ##################################
no, n1, n2 = output_efield.shape
n1_red = 31
psf_coverslip = np.zeros((no + n2, n1_red, n2))
for ii in range(n2):
    psf_coverslip[ii : ii + no, :, ii] = np.abs(output_efield[:, nxy//2 - n1_red//2:nxy//2 + n1_red//2 + 1, ii])**2
    # psf_coverslip[-(ii + 1 + no): -(ii + 1), :, ii] = np.abs(output_efield[:, nxy//2 - n1_red//2:nxy//2 + n1_red//2 + 1, ii])**2

psf_coverslip /= np.max(psf_coverslip)

# note that X and Y are swapped compared with my usual way of thinking about this...
# i0' -> X; i1' -> Y; i2' -> Z
dx = np.cos(theta) * dxy_out / total_mag
dy = dxy_out / total_mag
dz = np.sin(theta) * dxy_out / total_mag

xs_cs = np.arange(no + n2) * dx
xs_cs -= np.mean(xs_cs)
ys_cs = np.arange(n1_red) * dy
ys_cs -= np.mean(ys_cs)
zs_cs = np.arange(n2) * dz
zs_cs -= np.mean(zs_cs)

# put psf_coverslip in z, y, x order
psf_coverslip = psf_coverslip.transpose([2, 1, 0])

# in um
extent_xy_coverslip = [xs_cs[0] - 0.5 * dx, xs_cs[-1] + 0.5 * dx,
                       ys_cs[0] - 0.5 * dy, ys_cs[-1] + 0.5 * dy]
extent_xy_coverslip = [e / 1e-3 for e in extent_xy_coverslip]

extent_yz_coverslip = [ys_cs[0] - 0.5 * dy, ys_cs[-1] + 0.5 * dy,
                       zs_cs[0] - 0.5 * dz, zs_cs[-1] + 0.5 * dz]
extent_yz_coverslip = [e / 1e-3 for e in extent_yz_coverslip]

extent_xz_coverslip = [xs_cs[0] - 0.5 * dx, xs_cs[-1] + 0.5 * dx,
                       zs_cs[0] - 0.5 * dz, zs_cs[-1] + 0.5 * dz]
extent_xz_coverslip = [e / 1e-3 for e in extent_xz_coverslip]

# ##################################
# plot PSF
# ##################################
figh = plt.figure()
plt.suptitle("PSF, coverslip")

len_scale = 0.8

ax = plt.subplot(1, 3, 1)
ax.imshow(np.max(psf_coverslip, axis=0), aspect="equal", origin="lower", cmap="bone", interpolation="none",
          extent=extent_xy_coverslip)
ax.set_xlabel("X (um)")
ax.set_ylabel("Y (um)")
ax.set_xlim([-len_scale, len_scale])
ax.set_ylim([-len_scale, len_scale])

ax = plt.subplot(1, 3, 2)
ax.imshow(np.max(psf_coverslip, axis=1), aspect="equal", origin="lower", cmap="bone", interpolation="none",
          extent=extent_xz_coverslip)
ax.set_xlabel("X (um)")
ax.set_ylabel("Z (um)")
ax.set_xlim([-len_scale, len_scale])
ax.set_ylim([-len_scale, len_scale])

ax = plt.subplot(1, 3, 3)
ax.imshow(np.max(psf_coverslip, axis=2), aspect="equal", origin="lower", cmap="bone", interpolation="none",
          extent=extent_yz_coverslip)
ax.set_xlabel("Y (um)")
ax.set_ylabel("Z (um)")
ax.set_xlim([-len_scale, len_scale])
ax.set_ylim([-len_scale, len_scale])

# ##################################
# export results as tif
# ##################################
# tifffile.imwrite(Path(r"C:\Users\q2ilab\Desktop\snouty_psf.tif"),
#                  tifffile.transpose_axes(psf_coverslip.astype(np.float32), "ZYX", asaxes="TZCYXS"),
#                  imagej=True,
#                  resolution=(1 / (dx * 1e3), 1 / (dy * 1e3)),
#                  metadata={"Info": "snouty psf, theta = %0.2fdeg" % (theta * 180/np.pi),
#                            "unit": "um", "spacing": (dz * 1e3)})

