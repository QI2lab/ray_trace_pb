"""
Test cardinal points
"""
import numpy as np
import matplotlib.pyplot as plt
import raytrace.raytrace as rt
from raytrace.materials import Nsf11, Ebaf11, Vacuum, Constant

wlen = 0.635
nobj = 1.1
nimg = 1.333

system = rt.Doublet(Ebaf11(),
                    Nsf11(),
                    radius_crown=50.8,
                    radius_flint=-247.7,
                    radius_interface=-41.7,
                    thickness_crown=20.,
                    thickness_flint=3.,
                    aperture_radius=25.4,
                    input_collimated=False,
                    #names="AC508-075-A-ML"
                    )
fp1, fp2, pp1, pp2, np1, np2, efl1, efl2 = system.get_cardinal_points(wlen,
                                                                      Constant(nobj),
                                                                      Constant(nimg))
system = system.concatenate(rt.FlatSurface([0, 0, 0], [0, 0, 1], 25.4),
                            Constant(nimg),
                            fp2[-1] - system.surfaces[-1].paraxial_center[-1] + 10.
                            )

figh = plt.figure(figsize=(16, 9))
figh.suptitle(f"n_img = {nimg:.3f}, n_obj = {nobj:.3f}")
axs = [figh.add_subplot(2, 2, ii) for ii in range(1, 4)]
for ax in axs:
    ax.axvline(fp1[-1], c='r', label="fp1")
    ax.axvline(fp2[-1], c='r', label="fp2")
    ax.axvline(pp1[-1], c='b', label="pp1")
    ax.axvline(pp2[-1], c='b', label="pp2")
    ax.plot(np1[-1], np1[0], 'go', label="np1")
    ax.plot(np2[-1], np2[0], 'mo', label="np2")
    ax.legend()


# launch rays that appear to go through nodal point
rays_nodal = rt.get_ray_fan(np1, 3*np.pi/180, 21, wlen)
rays_in_nodal, _ = rt.propagate_ray2plane(rays_nodal, [0, 0, 1], [0, 0, fp1[-1]], Vacuum())
rays_out_nodal = system.ray_trace(rays_in_nodal, Constant(nobj), Constant(nimg))

rays_in_fp1 = rt.get_ray_fan(fp1, 2*np.pi/180, 21, wlen)
rays_out_fp1 = system.ray_trace(rays_in_fp1, Constant(nobj), Constant(nimg))

rays_in_coll = rt.get_collimated_rays(fp1, 10, 21, wlen)
rays_out_coll = system.ray_trace(rays_in_coll, Constant(nobj), Constant(nimg))

system.plot(rays_out_nodal, ax=axs[0])
axs[0].set_title("Nodal pt 1")
system.plot(rays_out_fp1, ax=axs[1])
axs[1].set_title("rays from focal pt 1")
system.plot(rays_out_coll, ax=axs[2])
axs[2].set_title("rays through focal pt 2")
