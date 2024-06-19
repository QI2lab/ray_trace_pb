"""
Ray tracing simulation using mirrors
"""
import matplotlib.pyplot as plt
import numpy as np
import raytrace.raytrace as rt
from raytrace.materials import Vacuum

rays = rt.get_ray_fan([0, 0, 0], 5*np.pi/180, 5, 0.785)
theta = np.pi/4 - np.pi/30
system = rt.System([rt.PlaneMirror([0, 0, 30], [-np.sin(theta), 0, -np.cos(theta)], 25),
                           rt.PlaneMirror([-50, 0, 30], [1 / np.sqrt(2), 0, 1 / np.sqrt(2)], 25),
                           rt.FlatSurface([-50, 0, 60], [0, 0, 1], 25)
                           ],
                   [Vacuum(), Vacuum()]
                   )

rays = system.ray_trace(rays, Vacuum(), Vacuum())
system.plot(rays)

plt.show()
