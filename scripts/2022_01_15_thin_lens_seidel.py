"""
Ray trace "thin lens" and calculate Seidel aberration coefficients
"""
import numpy as np
import matplotlib.pyplot as plt
import raytrace.raytrace as rt
from raytrace.materials import Vacuum, Constant

wavelength = 0.532
aperture_radius = 25.4
lens_start = 400
n = 1.5
r1 = 100
r2 = -200
t = -0.1
system = rt.System([rt.FlatSurface([0, 0, 0], [0, 0, 1], aperture_radius),
                    rt.FlatSurface([0, 0, lens_start + t], [0, 0, 1], aperture_radius),
                    rt.SphericalSurface.get_on_axis(r1, lens_start, aperture_radius),
                    rt.SphericalSurface.get_on_axis(r2, lens_start, aperture_radius),
                    ],
                   [Vacuum(), Vacuum(), Constant(n)]
                   )

# auto-focus
focus = system.auto_focus(wavelength, Vacuum(), Vacuum(), mode="paraxial-focused")

s1 = -lens_start
h = s1 / (t - s1)
H = t / 1
k = H / h
p = (n-1) * (1/r1  - 1/r2)
sig = (n-1) * (1/r1  + 1/r2)
K = -1/s1 - p/2
K1 = K + (sig + n * p) / (2*(n-1))
K2 = K + (sig - n * p) / (2*(n-1))
U = n**2 / (8*(n-1)**2) * p**3 - n / (2*(n+2)) * K**2 * p + 1 / (2*n*(n+2)) * p * ((n+2)/(2*(n-1)) * sig + 2 * (n+1) * K)**2
V = 1/ (2*n) * p * ((n+1) / (2*(n-1)) * sig + (2*n+1) * K)
B = h**4 * U
F = h**4 * k * U + h**2 * V
C = h**4 * k**2 * U + 2 * h**2 * k * V + 0.5 * p
D = h**4 * k**2 * U + 2 * h**2 * K * V + (n+1) / (2*n) * p
E = h**4 * k**3 * U + 3 * h**2 * k**2 * V + k * (3*n+1) / (2*n) * p
# C = (2*n + 1) / (4*n) * p - 1 / (4*n) * p
# D = (2*n + 1) / (4*n) * p + 1 / (4*n) * p

# entrance pupil coincides w lens we expect
ab_exp = np.array([B, F, C, D, E])

# test aberrations
abs = system.compute_third_order_seidel(wavelength)
for ii in range(abs.shape[0]):
    print(("Surface %02d: " + 5 * "%+0.5e, ") % ((ii,) + tuple(abs[ii])))
print("total     : " + 5 * "%+0.5e, " % tuple(np.sum(abs, axis=0)))

print("expected:")
print("total     : " + 5 * "%+0.5e, " % tuple(ab_exp))
