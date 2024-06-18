"""
Test chromatic shift for relay consisting of
ACT508-100-B

at 532nm and 785nm
"""
import matplotlib.pyplot as plt
import numpy as np
import raytrace.raytrace as rt
from raytrace.materials import Vacuum, Nlak22, Nsf6, Nsf6ht

w1 = 0.785 # um
w2 = 0.532
nrays = 101 # must be odd
offset = 5
radius = 25.4
beam_rad = 5

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

z180 = 10
# z100 = (t180c + t180f) + bfl180 + 100
z100 = z180 + (t180c + t180f) + 264.85
zend = z100 + 30

system = rt.System([# AC508-180-AB-ML. Infinity focus to the left
            rt.SphericalSurface.get_on_axis(r180c, z180, radius),
            rt.SphericalSurface.get_on_axis(r180i, z180 + t180c, radius),
            rt.SphericalSurface.get_on_axis(r180f, z180 + t180c + t180f, radius),
            # AC508-100-B-ML. Infinity focus to the right
            rt.SphericalSurface.get_on_axis(-r100f, z100, radius),
            rt.SphericalSurface.get_on_axis(-r100i, z100 + t100f, radius),
            rt.SphericalSurface.get_on_axis(-r100c, z100 + t100f + t100c, radius),
            # final focal plane
            rt.FlatSurface([0, 0, zend], [0, 0, 1], radius)
            ],
            [Nlak22(), Nsf6(), Vacuum(),  #AC508-180-AB-ML
            Nsf6ht(), Nlak22(), Vacuum(),  #AC508-100-B-ML
             ]
            )

# do ray tracing
rays1 = rt.get_collimated_rays([0, 0, 0], beam_rad, nrays, w1)
rays2 = rt.get_collimated_rays([0, 0, 0], beam_rad, nrays, w2)

rays1 = system.ray_trace(rays1, Vacuum(), Vacuum())
rays2 = system.ray_trace(rays2, Vacuum(), Vacuum())

figh, ax = system.plot(rays1, colors="r", label=f"{w1 * 1e3:.1f}nm", figsize=(16, 8))
system.plot(rays2, colors="g", label=f"{w2 * 1e3:.1f}nm", ax=ax)
ax.legend()
plt.show()



figh = plt.figure()
ax = figh.add_subplot(1, 1, 1)
ax.plot(rays1[-2, :, 0], np.arctan2(rays1[-2, :, 3], rays1[-2, :, 5]) * 180/np.pi, '-x', label=f"{w1 * 1e3:.1f}nm")
ax.plot(rays2[-2, :, 0], np.arctan2(rays2[-2, :, 3], rays2[-2, :, 5]) * 180/np.pi, '-x', label=f"{w2 * 1e3:.1f}nm")
ax.set_xlabel("height (mm)")
ax.set_ylabel("angle(deg)")
ax.legend()
