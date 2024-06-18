"""
Lightsheet with refocus using ETL
"""
import time
import datetime
import matplotlib.pyplot as plt
import numpy as np
import raytrace.raytrace as rt
from raytrace.materials import Constant
from pathlib import Path
import zarr

# save dir
tstamp = datetime.datetime.now().strftime('%Y_%m_%d_%H;%M;%S')

fdir = Path(r"/mnt/scratch1/2024_04_01") / f"{tstamp:s}_lightsheet_raytrace"
fdir.mkdir(parents=True, exist_ok=True)

save_dir_trains = fdir / "rt_diagrams"
save_dir_trains.mkdir(exist_ok=True)

# define optical train
settings = {"nrays": 1001,
            "wavelength": 0.532,
            "aperture_radius_etl": 8,
            "aperture_radius": 50.8 / 2, # mm,
            "n_etl": 1.3,
            "t_edge": 5,
            "f1": 160, # relay lens 1
            "f2": 120,
            "fobj": 20,
            "t_coverglass": 1.25, # coverglass
            "n_coverglass": 1.4585,
            "dz_coverglass": 10,
            "n_immersion": 1.333
            }
n_surfaces = 17

rad_curvs1 = np.linspace(settings["aperture_radius_etl"], 55, 201)
rad_curvs2 = np.linspace(55, 400, 101, endpoint=True)
rad_curvs = np.concatenate((rad_curvs1, rad_curvs2, np.array([1e9])))

def f2d(f): return 1e3 / f
def d2f(d): return 1e3 / d

focal_lens_mm = rad_curvs / (settings["n_etl"] - 1)
dpt = f2d(focal_lens_mm)

intersect_spread = np.zeros_like(rad_curvs)

# save ray results to zarr
z = zarr.open(fdir / f"rays.zarr", "w")
z.create("rays",
         shape=(len(rad_curvs), n_surfaces, settings["nrays"], 8),
         chunks=(1, n_surfaces, settings["nrays"], 8),
         dtype=float)
z.array("radius_curvatures", rad_curvs, dtype=float)
z.array("etl_diopters", dpt, dtype=float)
z.array("focal_lens_mm", focal_lens_mm, dtype=float)
z.rays.attrs["array_columns"] = ["x", "y", "z", "dx", "dy", "dz", "phase", "wavelength"]
z.attrs["settings"] = settings

# ray trace optical trains with different ETL curvature
plt.switch_backend("Agg")
tstart = time.perf_counter()
for ii, rad_curv in enumerate(rad_curvs):
    print(f"{ii+1:d}/{len(rad_curvs):d} in {time.perf_counter() - tstart:.2f}s", end="\r")
    # etl derived params
    t_center = settings["t_edge"] + rad_curv * (1 - np.sqrt(1 - (settings["aperture_radius_etl"] / rad_curv) ** 2))



    rays_start = rt.get_collimated_rays([0, 0, -1],
                                        8,
                                        settings["nrays"],
                                        settings["wavelength"]
                                        )

    etl = rt.System([rt.FlatSurface([0, 0, 0],
                                    [0, 0, 1],
                                    settings["aperture_radius_etl"]),
                     rt.SphericalSurface.get_on_axis(-rad_curv,
                                                     t_center,
                                                     settings["aperture_radius_etl"]),
                     ],
                    materials=[Constant(settings["n_etl"])],
                    names="etl")

    l1 = rt.System([rt.PerfectLens(settings["f1"],
                                   [0, 0, 0],
                                   [0, 0, 1],
                                   alpha=np.arcsin(0.1))],
                   [],
                   names="l1"
                   )

    l2 = rt.System([rt.PerfectLens(settings["f2"],
                                   [0, 0, 0],
                                   [0, 0, 1],
                                   alpha=np.arcsin(0.1))],
                   [],
                   names="l2"
                   )

    obj = rt.System([rt.PerfectLens(settings["fobj"],
                                    [0, 0, 0],
                                    [0, 0, 1],
                                    alpha=np.arcsin(0.3))],
                    [],
                    names="obj"
                    )

    cglass = rt.System([rt.FlatSurface([0, 0, 0],
                                       [0, 0, 1],
                                       settings["aperture_radius"]),
                        rt.FlatSurface([0, 0, settings["t_coverglass"]],
                                       [0, 0, 1],
                                       settings["aperture_radius"]),
                        rt.FlatSurface([0, 0, 30],
                                       [0, 0, 1],
                                       settings["aperture_radius"])
                        ],
                       [Constant(settings["n_coverglass"]),
                        Constant(settings["n_immersion"])],
                       "coverglass")


    osys = etl.concatenate(l1, rt.Vacuum(), settings["f1"] - (t_center - settings["t_edge"]))
    osys = osys.concatenate(l2, rt.Vacuum(), settings["f1"] + settings["f2"])
    osys = osys.concatenate(obj, rt.Vacuum(), settings["f2"] + settings["fobj"])
    osys = osys.concatenate(cglass, rt.Vacuum(), settings["dz_coverglass"])

    # ray trace
    rays = osys.ray_trace(rays_start, rt.Vacuum(), rt.Vacuum())
    z.rays[ii] = rays

    # plot optical System
    desc_str = f"f={focal_lens_mm[ii]:.1f}mm_r={rad_curv:.1f}mm_dpt={dpt[ii]:.1f}"
    figh, ax = osys.plot(rays, figsize=(20, 15))
    figh.suptitle(desc_str)
    ax.set_xlim([-100, 700])

    fname_fig = save_dir_trains / f"{desc_str:s}_rays.png"
    figh.savefig(fname_fig)
    plt.close(figh)

    # focus versus height
    ray_intersections = rt.intersect_rays(rays[-1], np.array([0, 0, 0, 0, 0, 1])[None, :])
    h = rays[0, :, 0]
    intersects = ray_intersections[:, 2]
    intersect_spread[ii] = np.nanmax(intersects) - np.nanmin(intersects)

    # figh2 = plt.figure(figsize=(20, 15))
    # figh2.suptitle(f"{desc_str:s} spherical aberration")
    # ax = figh2.add_subplot(1, 1, 1)
    # ax.plot(h, z)
    # ax.set_xlabel("height (mm)")
    # ax.set_ylabel("optical axis intersection position (mm)")
    #
    # figh2.savefig(fdir / f"spherical_ab_{desc_str:s}_rays.png")
    # plt.close(figh2)

print()

plt.switch_backend("Qt5Agg")
xlim = [np.min(focal_lens_mm), np.max(focal_lens_mm[:-2])]
xticks = np.arange(50, xlim[1], 100)

figh_out = plt.figure(figsize=(20, 15))
ax = figh_out.add_subplot(2, 1, 1)
ax.set_ylabel("spread of intersects (mm)")
ax.plot(focal_lens_mm, intersect_spread)
ax.set_xlim(xlim)
ax.set_xticks(xticks)
secax = ax.secondary_xaxis('top', functions=(f2d, d2f))
secax.set_xlabel('Power (dpt)')

notnan_fraction = 1 - np.sum(np.isnan(z.rays[:, -1, :, 0]), axis=-1) / settings["nrays"]
ax = figh_out.add_subplot(2, 1, 2)
ax.set_ylabel("Fraction of rays transmitted")
ax.set_xlabel("Focal length (mm)")
ax.plot(focal_lens_mm, notnan_fraction)
ax.set_xlim(xlim)
ax.set_xticks(xticks)

secax.set_xticks(f2d(xticks))

figh_out.savefig(fdir / f"summary.png")