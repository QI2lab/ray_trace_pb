"""
ray represented by eight parameters
(xo, yo, zo, dx, dy, dz, phase, wavelength) where (xo, yo, zo) is a point the ray passes through,
and (dx, dy, dz) is a unit vector specifying its direction. The ray travels along the line
C(t) = (xo, yo, zo) + t * (dx, dy, dz)

A paraxial ray has a different representation, in terms of height and angle,
(hx, n*ux, hy, n*uy). Heights and angles are defined relative to some point and axis,
(xo, yo, zo) and (dx, dy, dz).

We adopt a default coordinate system where z points to the right, along the optical axis. x points upwards,
and y points out of the plane, ensuring the coordinate System is right-handed
"""
from typing import Optional, Union
from collections.abc import Sequence
from numpy.typing import NDArray
import copy
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
import matplotlib.pyplot as plt
import warnings
from raytrace.materials import Material, Vacuum

try:
    import cupy as cp
    array = Union[NDArray, cp.ndarray]
except ImportError:
    cp = None
    array = NDArray


def get_free_space_abcd(d: float, n: float = 1.) -> NDArray:
    """
    Compute the ray-transfer (ABCD) matrix for free space beam propagation

    :param d: distance beam propagated
    :param n: index of refraction
    :return mat:
    """
    mat = np.array([[1, d/n], [0, 1]])
    return mat


# tools for creating ray fans, manipulating rays, etc.
def get_ray_fan(pt: NDArray,
                theta_max: float,
                n_thetas: int,
                wavelengths: float,
                nphis: int = 1,
                center_ray=(0, 0, 1)) -> array:
    """
    Get fan of rays emanating from pt

    :param pt: [cx, cy, cz]
    :param theta_max: maximum angle in radians
    :param n_thetas: number of rays at different angles on axis
    :param wavelengths:
    :param nphis: number of points of rotation about the optical axis. If nphis = 1, all rays will be in the plane
    :param center_ray:
    """

    xp = cp if cp is not None and isinstance(pt, cp.ndarray) else np

    # consider the central ray in direction no. Construct an orthonormal basis from enx = y x no, eny = no x enx
    # then construct rays v(theta, phi), where theta is the angle between v and no and phi is the angle rotated about no
    # v(theta, phi) = cos(theta) * no + cos(phi) * sin(theta) * enx + sin(phi) * sin(theta) * eny
    center_ray = np.array(center_ray)
    if np.linalg.norm(center_ray) != 1:
        raise ValueError("center_ray must be a unit vector")

    thetas = np.linspace(-theta_max, theta_max, n_thetas)
    phis = np.arange(nphis) * 2*np.pi / nphis
    rays = np.zeros((n_thetas * nphis, 8))

    tts, pps = np.meshgrid(thetas, phis)
    tts = tts.ravel()
    pps = pps.ravel()

    enx = np.cross(np.array([0, 1, 0]), center_ray)
    enx = enx / np.linalg.norm(enx)
    eny = np.cross(center_ray, enx)

    pt = np.array(pt).squeeze()

    rays[:, 0] = pt[0]
    rays[:, 1] = pt[1]
    rays[:, 2] = pt[2]

    rays[:, 3] = center_ray[0] * np.cos(tts) + enx[0] * np.cos(pps) * np.sin(tts) + eny[0] * np.sin(pps) * np.sin(tts)
    rays[:, 4] = center_ray[1] * np.cos(tts) + enx[1] * np.cos(pps) * np.sin(tts) + eny[1] * np.sin(pps) * np.sin(tts)
    rays[:, 5] = center_ray[2] * np.cos(tts) + enx[2] * np.cos(pps) * np.sin(tts) + eny[2] * np.sin(pps) * np.sin(tts)

    rays[:, 6] = 0
    rays[:, 7] = wavelengths

    return rays


def get_collimated_rays(pt: NDArray,
                        displacement_max,
                        n_disps: int,
                        wavelengths: float,
                        nphis: int = 1,
                        phi_start: float = 0.,
                        normal=(0, 0, 1)) -> NDArray:
    """
    Get a fan of collimated arrays along a certain direction. The rays will be generated in a plane
    with normal along this direction, which will generally not be perpendicular to the optical axis.

    Note that this approach avoids the need to know what the index of refraction of the medium is

    :param pt: point in the origin plane
    :param displacement_max: maximum radial displacement
    :param n_disps: number of displacements
    :param wavelengths: either floating point or an array the same size as n_disps * nphis
    :param nphis: number of rays in azimuthal direction
    :param phi_start: angle about normal to start at
    :param normal: normal of plane
    :return rays:
    """
    if np.abs(np.linalg.norm(normal) - 1) > 1e-12:
        raise ValueError("normal must be a normalized vector")

    # build all angles and offsets and put in 1d arrays
    phis = np.arange(nphis) * 2*np.pi / nphis + phi_start
    offs = np.linspace(-displacement_max, displacement_max, n_disps)
    pps, oos = np.meshgrid(phis, offs)
    pps = pps.ravel()
    oos = oos.ravel()

    pt = np.array(pt).squeeze()
    normal = np.array(normal).squeeze()
    # build orthogonal unit vectors versus the normal
    # n1 -> ex if normal = [0, 0, 1]
    n1 = np.cross(np.array([0, 1, 0]), normal)
    # except in the case normal is already [0, 1, 0]
    if np.linalg.norm(n1) == 0:
        n1 = np.cross(normal, np.array([1, 0, 0]))

    n1 = n1 / np.linalg.norm(n1)

    # n2 -> ey if normal = [0, 0, 1]
    n2 = np.cross(normal, n1)
    n2 = n2 / np.linalg.norm(n2)

    # construct rays
    rays = np.zeros((n_disps * nphis, 8))
    # position = d * (n1 * cos(theta) + n2 * sin(theta))
    rays[:, 0:3] = np.expand_dims(pt, axis=0) + \
                   np.expand_dims(n1, axis=0) * np.expand_dims(oos * np.cos(pps), axis=1) + \
                   np.expand_dims(n2, axis=0) * np.expand_dims(oos * np.sin(pps), axis=1)
    # rays are parallel
    rays[:, 3] = normal[0]
    rays[:, 4] = normal[1]
    rays[:, 5] = normal[2]
    # assume phase is the same on plane perpendicular to the normal
    rays[:, 6] = 0
    # wavelength
    rays[:, 7] = wavelengths

    return rays


def intersect_rays(ray1: array, ray2: array) -> array:
    """
    Find intersection point between two rays, assuming free space propagation. If either s or t is negative
    then these rays previously intersected

    :param ray1:
    :param ray2:
    :return intersection_pt:
    """
    xp = cp if cp is not None and isinstance(ray1, cp.ndarray) else np

    ray1 = xp.atleast_2d(ray1)
    ray2 = xp.atleast_2d(ray2)

    if len(ray1) == 1 and len(ray2) > 1:
        ray1 = xp.tile(ray1, (len(ray2), 1))

    if len(ray2) == 1 and len(ray1) > 1:
        ray2 = xp.tile(ray2, (len(ray1), 1))

    if len(ray1) != len(ray2):
        raise ValueError("ray1 and ray2 must be the same length")

    # ray1 = (x1, y1, z1) + t * (dx1, dy1, dz1)
    # ray2 = (x2, y2, z2) + s * (dx2, dy2, dz2)
    x1 = ray1[:, 0]
    y1 = ray1[:, 1]
    z1 = ray1[:, 2]
    dx1 = ray1[:, 3]
    dy1 = ray1[:, 4]
    dz1 = ray1[:, 5]

    x2 = ray2[:, 0]
    y2 = ray2[:, 1]
    z2 = ray2[:, 2]
    dx2 = ray2[:, 3]
    dy2 = ray2[:, 4]
    dz2 = ray2[:, 5]

    # intersection problem is overdetermined, so this is solution if there is one
    # determine distance along ray1
    s = xp.zeros(len(ray1)) * np.nan
    with np.errstate(invalid="ignore"):
        use_xz = dx2 * dz1 - dz2 * dx1 != 0
        use_xy = xp.logical_and(xp.logical_not(use_xz),
                                dx2 * dy1 - dy2 * dx1)
        use_yz = xp.logical_and.reduce((xp.logical_not(use_xz),
                                        xp.logical_not(use_xy),
                                        dz2 * dy1 - dy2 * dz1))

        s[use_xz] = (((z2 - z1) * dx1 - (x2 - x1) * dz1) / (dx2 * dz1 - dz2 * dx1))[use_xz]
        s[use_xy] = (((y2 - y1) * dx1 - (x2 - x1) * dy1) / (dx2 * dy1 - dy2 * dx1))[use_xy]
        s[use_yz] = (((y2 - y1) * dz1 - (z2 - z1) * dy1) / (dz2 * dy1 - dy2 * dz1))[use_yz]
        # otherwise, d1 \cross d2 = 0, so rays are parallel and leave as NaN

    # determine distance along ray2, but avoid undefined expressions
    t = xp.zeros(len(ray1)) * np.nan
    with np.errstate(all="ignore"):
        use_z = dz1 != 0
        use_y = xp.logical_and(xp.logical_not(use_z), dy1 != 0)
        # otherwise dx1 is guaranteed to be != 0 since dr1 is a unit vector
        use_x = xp.logical_not(xp.logical_or(use_z, use_y))
        t[use_z] = ((z2 + s * dz2 - z1) / dz1)[use_z]
        t[use_y] = ((y2 + s * dy2 - y1) / dy1)[use_y]
        t[use_x] = ((x2 + s * dx2 - x1) / dx1)[use_x]

    # but to verify the solution, must check intersection points are actually equal
    intersect1 = xp.stack((x1, y1, z1), axis=1) + xp.expand_dims(t, axis=1) * xp.stack((dx1, dy1, dz1), axis=1)
    intersect2 = xp.stack((x2, y2, z2), axis=1) + xp.expand_dims(s, axis=1) * xp.stack((dx2, dy2, dz2), axis=1)

    with np.errstate(invalid="ignore"):
        not_sol = xp.max(xp.abs(intersect1 - intersect2), axis=1) > 1e-12
        intersect1[not_sol] = np.nan

    return intersect1


def propagate_ray2plane(rays: array,
                        normal: array,
                        center: array,
                        material: Material,
                        exclude_backward_propagation: bool = False) -> (NDArray, NDArray):
    """
    Find intersection between rays and a plane. Plane is defined by a normal vector and a point on the
    plane

    :param rays: N x 8 array
    :param normal: normal of the plane. Should be broadcastable to the shape N x 3
    :param center: point on the plane. Should be broadcastable to the shape N x 3
    :param material: Material through which rays are propagating
    :param exclude_backward_propagation:
    :return rays_out, ts: where rays_out is an N x 8 array and ts is a length N array giving the propagation distance
    """
    xp = cp if cp is not None and isinstance(rays, cp.ndarray) else np
    normal = xp.asarray(normal)
    center = xp.asarray(center)

    rays = xp.atleast_2d(xp.array(rays, copy=True))

    normal = xp.array(normal).squeeze()
    if normal.ndim == 1:
        normal = xp.expand_dims(normal, axis=0)

    center = xp.array(center).squeeze()
    if center.ndim == 1:
        center = xp.expand_dims(center, axis=0)

    xo = rays[:, 0]
    yo = rays[:, 1]
    zo = rays[:, 2]
    dx = rays[:, 3]
    dy = rays[:, 4]
    dz = rays[:, 5]
    phase_o = rays[:, 6]
    wls = rays[:, 7]
    xc = center[:, 0]
    yc = center[:, 1]
    zc = center[:, 2]
    nx = normal[:, 0]
    ny = normal[:, 1]
    nz = normal[:, 2]

    # parameterize distance along ray by t
    ts = - ((xo - xc) * nx + (yo - yc) * ny + (zo - zc) * nz) / (dx * nx + dy * ny + dz * nz)

    # determine if this is a forward or backward propagation for the ray
    with np.errstate(invalid="ignore"):
        prop_direction = xp.ones(rays.shape[0], dtype=int)
        prop_direction[ts < 0] = -1

    # find intersection points
    prop_dist_vect = xp.stack((dx, dy, dz), axis=1) * xp.expand_dims(ts, axis=1)
    pts = xp.stack((xo, yo, zo), axis=1) + prop_dist_vect
    phase_shift = xp.linalg.norm(prop_dist_vect, axis=1) * prop_direction * 2 * np.pi / wls * material.n(wls)

    # assemble output rays
    rays_out = xp.concatenate((pts, xp.stack((dx, dy, dz, phase_o + phase_shift, wls), axis=1)), axis=1)

    # replace back propagating rays with Nans if desired
    if exclude_backward_propagation:
        rays_out[prop_direction == -1, :] = xp.nan

    return rays_out, ts


def ray_angle_about_axis(rays: array, reference_axis: array) -> (array, array):
    """
    Given a set of rays, compute their angles relative to a given axis, and compute the orthogonal direction to the
    axis which the ray travels in

    :param rays:
    :param reference_axis:
    :return angles, na:
    """
    xp = cp if cp is not None and isinstance(rays, cp.ndarray) else np

    rays = xp.atleast_2d(rays)
    reference_axis = xp.asarray(reference_axis)

    cosines = xp.sum(rays[:, 3:6] * xp.expand_dims(reference_axis, axis=0), axis=1)
    angles = xp.arccos(cosines)
    na = rays[:, 3:6] - xp.expand_dims(cosines, axis=1) * xp.expand_dims(reference_axis, axis=0)
    na = na / xp.expand_dims(np.linalg.norm(na, axis=1), axis=1)

    return angles, na


def dist_pt2plane(pts: array,
                  normal: array,
                  center: array) -> (array, array):
    """
    Calculate minimum distance between points and a plane defined by normal and center

    :param pts:
    :param normal:
    :param center:
    :return dists, nearest_pts:
    """
    xp = cp if cp is not None and isinstance(pts, cp.ndarray) else np

    pts = xp.atleast_2d(pts)
    npts = pts.shape[0]

    rays = xp.concatenate((pts, xp.tile(normal, (npts, 1)), xp.zeros((npts, 2))), axis=1)
    rays_int, _ = propagate_ray2plane(rays, normal, center, Vacuum())

    dists = xp.linalg.norm(rays_int[:, :3] - pts, axis=1)
    nearest_pts = rays_int[:, :3]

    return dists, nearest_pts


# ################################################
# collections of optical elements
# ################################################
class System:
    """
    Collection of optical surfaces
    """
    # todo: think life will be easier if keep materials at start and end too

    def __init__(self,
                 surfaces: list,
                 materials: list[Material],
                 names: list[str] = None,
                 surfaces_by_name=None,
                 aperture_stop: Optional[int] = None):
        """

        :param surfaces: length n
        :param materials: length n-1
        :param names:
        :param surfaces_by_name:
        """
        if len(materials) > 1:
            if len(materials) != (len(surfaces) - 1):
                raise ValueError(f"len(materials) = {len(materials):d} != len(surfaces) - 1 = {len(surfaces) - 1:d}")

        self.surfaces = surfaces
        self.materials = materials
        self.aperture_stop = aperture_stop

        if names is None:
            self.names = [""]
        else:
            if not isinstance(names, list):
                names = [names]
            self.names = names

        # should be able to get name of surfaces ii from self.names[self.surfaces_by_name[ii]]
        if surfaces_by_name is None:
            self.surfaces_by_name = np.zeros(len(surfaces), dtype=int)
        else:
            if len(surfaces_by_name) != len(surfaces):
                raise ValueError("len(surfaces_by_name) must equal len(surfaces)")

            self.surfaces_by_name = np.array(surfaces_by_name).astype(int)

    def reverse(self):
        """
        flip direction of the optic we are considering (so typically rays now enter from the right)
        :return:
        """
        surfaces_rev = [copy.deepcopy(self.surfaces[-ii]) for ii in range(1, len(self.surfaces) + 1)]

        for ii in range(len(self.surfaces)):
            surfaces_rev[ii].input_axis *= -1
            surfaces_rev[ii].output_axis *= -1

        materials_rev = [self.materials[-ii] for ii in range(1, len(self.materials) + 1)]

        return System(surfaces_rev, materials_rev)

    def concatenate(self,
                    other,
                    material: Material,
                    distance: Optional[float] = None,
                    axis: Sequence[float, float, float] = (0., 0., 1.)):
        """
        add another optical Surface to the end of this System

        :param other: the optical elements to add. Should be a System or a Surface
        :param material: the Material between the end of this System and the start of the next
        :param distance: the distance between the last surface of the current system and the first surface of the
         next system, determined by the paraxial center. If None, then place the next system at the
         coordinates of the surface
        :param axis:
        :return new_system:
        """
        # todo: want to make it possible to add surface at intermediate location
        # todo: how to make type hints work?

        # specify distance between surfaces as distances between the paraxial foci
        if isinstance(other, System):
            new_surfaces = [copy.deepcopy(s) for s in other.surfaces]
            new_materials = other.materials
            other_stop = other.aperture_stop
            new_surfaces_by_name = other.surfaces_by_name
            new_names = other.names
        elif isinstance(other, Surface):
            new_surfaces = [copy.deepcopy(other)]
            new_materials = []
            other_stop = None
            new_surfaces_by_name = np.array([0])
            new_names = [""]
        else:
            raise TypeError(f"other should be of type System or Surface, but was {type(other)}")

        if distance is not None:
            for ii, s in enumerate(new_surfaces):
                # C_i(new) = C_{i-1}(new) + [C_i(old) - C_{i-1}(old)]
                if ii == 0:
                    shift = self.surfaces[-1].paraxial_center + distance * np.array(axis) - s.paraxial_center
                else:
                    shift = new_surfaces[ii - 1].paraxial_center - other.surfaces[ii - 1].paraxial_center

                s.center += shift
                s.paraxial_center += shift

        surfaces_by_name = np.concatenate((self.surfaces_by_name,
                                           new_surfaces_by_name + np.max(self.surfaces_by_name) + 1))

        if self.aperture_stop is None:
            if other_stop is None:
                aperture_stop = self.aperture_stop
            else:
                aperture_stop = other_stop + len(self.surfaces)
        else:
            aperture_stop = self.aperture_stop

        return System(self.surfaces + new_surfaces,
                      self.materials + [material] + new_materials,
                      names=self.names + new_names,
                      surfaces_by_name=surfaces_by_name,
                      aperture_stop=aperture_stop)

    def set_aperture_stop(self,
                          surface_index: int):
        self.aperture_stop = surface_index

    def seidel_third_order(self,
                           wavelength: float,
                           initial_material: Material,
                           final_material: Material,
                           print_results: bool = False,
                           object_distance: float = 0.,
                           object_height: float = 0.,
                           object_angle: float = 0.,
                           ):
        """
        Calculate Seidel aberration coefficients
        We assume the initial object is at the first surface (if this is not true, add a surface)

        :param wavelength:
        :param initial_material:
        :param final_material:
        :param print_results:
        :param object_distance: distance of object before first surface. Positive if before surface
        :param object_height: field-of-view, used to calculate chief ray. Object height is used if object_distance
          is not infinite
        :param object_angle: field of view angle. object_angle is only used if object_distance = np.inf
        :return:
        """

        if self.aperture_stop is None:
            raise ValueError("aperture_stop was None, but aperture_stop must be provided to "
                             "compute Seidel aberrations")

        materials = [initial_material] + self.materials + [final_material]
        ns = np.array([m.n(wavelength) for m in materials])

        # get ray transfer matrices for each surface
        rt_mats = self.get_ray_transfer_matrix(wavelength, initial_material, final_material)
        rt_stop = rt_mats[self.aperture_stop]

        # compute marginal and chiefs rays at first surface
        if np.isinf(object_distance):
            h_chief_first = 0.
            u_chief_first = object_angle

            h_first = self.surfaces[self.aperture_stop].aperture_rad / rt_stop[0, 0]
            u_first = 0.
        else:
            rt_obj2stop = rt_stop.dot(get_free_space_abcd(object_distance, ns[0]))
            # B * n_start * u_start = h_stop
            h_start = 0.
            u_start = self.surfaces[self.aperture_stop].aperture_rad / rt_obj2stop[0, 1] / ns[0]
            h_first = rt_obj2stop[0, 0] * h_start + rt_obj2stop[0, 1] * ns[0] * u_start
            u_first = rt_obj2stop[1, 0] * h_start + rt_obj2stop[1, 1] * ns[0] * u_start

            h_chief_start = object_height
            u_chief_start = -rt_obj2stop[0, 0] / rt_obj2stop[0, 1] / ns[0] * h_chief_start  # A*h + B*n*u = h_chief = 0
            h_chief_first = rt_obj2stop[0, 0] * h_chief_start + rt_obj2stop[0, 1] * ns[0] * u_chief_start
            u_chief_first = rt_obj2stop[1, 0] * h_chief_start + rt_obj2stop[1, 1] * ns[0] * u_chief_start

        # trace marginal and chief rays
        # nsurfaces x 2 x 2 array
        # rays[:, :, 0] are marginal ray data (h, n*u); rays[:, :, 1] are chief ray data
        rays_start = np.array([[h_first, h_chief_first],
                               [ns[0] * u_first, ns[0] * u_chief_first]])
        rays = rt_mats.dot(rays_start)

        # values needed to calculate aberrations
        cs = np.array([1 / s.radius if isinstance(s, SphericalSurface) else 0 for s in self.surfaces])
        refraction_inv = ns[:-1] * rays[:-1, 0, 0] * cs + rays[:-1, 1, 0]
        refraction_inv_chief = ns[:-1] * rays[:-1, 0, 1] * cs + rays[:-1, 1, 1]
        delta_un = rays[1:, 1, 0] / ns[1:] / ns[1:] - rays[:-1, 1, 0] / ns[:-1] / ns[:-1]
        lagrange_inv = ns[:-1] * (rays[:-1, 0, 1] * rays[:-1, 1, 0] / ns[:-1] -
                                  rays[:-1, 0, 0] * rays[:-1, 1, 1] / ns[:-1])

        # compute aberrations following "Fundamentals of Optical Design" by Michael J. Kidger,
        # chapter 6, eqs 6.27-6.30 and 6.37
        # spherical, coma, astigmatism, field curvature, distortion
        aberrations = np.zeros((len(self.surfaces), 5)) * np.nan
        aberrations[:, 0] = -refraction_inv**2 * rays[:-1, 0, 0] * delta_un
        aberrations[:, 1] = -refraction_inv * refraction_inv_chief * rays[:-1, 0, 0] * delta_un
        aberrations[:, 2] = -refraction_inv_chief ** 2 * rays[:-1, 0, 0] * delta_un
        aberrations[:, 3] = -lagrange_inv ** 2 * cs * (1 / ns[1:] - 1 / ns[:-1])
        # aberrations[:, 4] = refraction_inv_chief / refraction_inv * (aberrations[:, 2] + aberrations[:, 3])
        aberrations[:, 4] = (-refraction_inv_chief ** 3 * rays[:-1, 0, 0] * (1 / ns[1:]**2 - 1 / ns[:-1]**2) +
                             rays[:-1, 0, 1] * refraction_inv_chief * cs *
                             (2 * rays[:-1, 0, 0] * refraction_inv_chief - rays[:-1, 0, 1] * refraction_inv) *
                             (1 / ns[1:] - 1 / ns[:-1])
                             )

        if print_results:
            print("surface,"
                  "          h,"
                  "          u,"
                  "       hbar,"
                  "       ubar,"
                  "   delta(u/n)"
                  "          A,"
                  "       Abar,"
                  "   Lag. inv."
                  )
            for ii in range(len(self.surfaces)):
                print(f"{ii:02d}:      "
                      f"{rays[ii, 0, 0]:10.6g}, "
                      f"{rays[ii, 1, 0] / ns[ii]:10.6g}, "
                      f"{rays[ii, 0, 1]:10.6g}, "
                      f"{rays[ii, 1, 1] / ns[ii]:10.6g}, "
                      f"{delta_un[ii]:10.6g}, "
                      f"{refraction_inv[ii]:10.6g}, "
                      f"{refraction_inv_chief[ii]:10.6g}, "
                      f"{lagrange_inv[ii]:10.6g}"
                      )


            print("surfaces,"
                  " spherical,"
                  "       coma,"
                  "     astig.,"
                  "   field curv.,"
                  "   distortion")
            for ii in range(len(self.surfaces)):
                print(f"{ii:02d}:      "
                      f"{aberrations[ii, 0]:10.6g}, "
                      f"{aberrations[ii, 1]:10.6g}, "
                      f"{aberrations[ii, 2]:10.6g}, "
                      f"{aberrations[ii, 3]:10.6g}, "
                      f"{aberrations[ii, 4]:10.6g}")
            print(f"sum:     "
                  f"{np.sum(aberrations[:, 0], axis=0):10.6g}, "
                  f"{np.sum(aberrations[:, 1], axis=0):10.6g}, "
                  f"{np.sum(aberrations[:, 2], axis=0):10.6g}, "
                  f"{np.sum(aberrations[:, 3], axis=0):10.6g}, "
                  f"{np.sum(aberrations[:, 4], axis=0):10.6g}")

        return aberrations

    def find_paraxial_collimated_distance(self,
                                          other,
                                          wavelength: float,
                                          initial_material: Material,
                                          intermediate_material: Material,
                                          final_material: Material,
                                          axis=None) -> (float, float):
        """
        Given two sets of surfaces (e.g. two lenses) determine the distance which should be inserted between them
        to give a System which converts collimated rays to collimated rays

        :param other:
        :param wavelength:
        :param initial_material:
        :param intermediate_material:
        :param final_material:
        :param axis:
        :return dx, dy:
        """
        mat1 = self.get_ray_transfer_matrix(wavelength, initial_material, intermediate_material)[-1]
        mat2 = other.get_ray_transfer_matrix(wavelength, intermediate_material, final_material)[-1]

        # todo: implement axis
        d = -(mat1[0, 0] / mat1[1, 0] + mat2[1, 1] / mat2[1, 0]) * intermediate_material.n(wavelength)
        return d

    def ray_trace(self,
                  rays: array,
                  initial_material: Material,
                  final_material: Material) -> array:
        """
        ray trace through optical System

        :param rays:
        :param initial_material:
        :param final_material:
        :return rays:
        """
        materials = [initial_material] + self.materials + [final_material]

        if len(materials) != len(self.surfaces) + 1:
            raise ValueError("length of materials should be len(surfaces) + 1")

        for ii in range(len(self.surfaces)):
            rays = self.surfaces[ii].propagate(rays, materials[ii], materials[ii + 1])

        return rays

    def gaussian_paraxial(self,
                          q_in: complex,
                          wavelength: float,
                          initial_material: Material,
                          final_material: Material,
                          print_results: bool = False):
        """

        :param q_in:
        :param wavelength:
        :param initial_material:
        :param final_material:
        :param print_results:
        :return:
        """

        ns = np.zeros(len(self.surfaces) + 1)
        qs = np.zeros(len(self.surfaces) + 1, dtype=complex)
        qs[0] = q_in
        for ii, s in enumerate(self.surfaces):
            if ii == 0:
                n1 = initial_material.n(wavelength)
            else:
                n1 = self.materials[ii - 1].n(wavelength)

            if ii < len(self.surfaces) - 1:
                n2 = self.materials[ii].n(wavelength)
                d = np.linalg.norm(self.surfaces[ii + 1].paraxial_center - s.paraxial_center)
            else:
                n2 = final_material.n(wavelength)
                d = 0.

            abcd = get_free_space_abcd(d, n2).dot(s.get_ray_transfer_matrix(n1, n2))
            qs[ii + 1] = (qs[ii] * abcd[0, 0] + abcd[0, 1]) / (qs[ii] * abcd[1, 0] + abcd[1, 1])
            ns[ii] = n1
            ns[ii + 1] = n2

        if print_results:
            import mcsim.analysis.gauss_beam as gb
            r, w_sqr, wo_sqr, z, zr = gb.q2beam_params(qs, wavelength, ns)

            print("surfaces \t R,"
                  "          w,"
                  "         wo,"
                  "          z,"
                  "          zr")
            for ii in range(len(self.surfaces) + 1):
                print(f"{ii:02d}: "
                      f"{r[ii]:10.6g}, "
                      f"{np.sqrt(w_sqr[ii]):10.6g}, "
                      f"{np.sqrt(wo_sqr[ii]):10.6g}, "
                      f"{z[ii]:10.6g}, "
                      f"{zr[ii]:10.6g}")

        return qs

    def get_ray_transfer_matrix(self,
                                wavelength: float,
                                initial_material: Material,
                                final_material: Material,
                                axis=None):
        """
        Generate the ray transfer (ABCD) matrices throughout an optical system.
        If the optical system as n surfaces, then this returns an array of size n+1 x 2 x 2,
        where the first n matrices transfer a ray to just before each surface, and the last matrix
        transfers a ray to just after the last surface

        :param wavelength:
        :param initial_material:
        :param final_material:
        :param axis: axis which should be a direction orthogonal to the main beam direction. Only relevant for
         non-symmetric optics
        :return abcd_matrix:
        """
        materials = [initial_material] + self.materials + [final_material]
        ns = np.array([m.n(wavelength) for m in materials])
        rt_mats = np.zeros((len(self.surfaces) + 1, 2, 2))
        for ii in range(len(self.surfaces) + 1):
            if ii == 0:
                rt_mats[ii] = get_free_space_abcd(0, ns[0])
            elif ii == len(self.surfaces):
                rt_next = self.surfaces[-1].get_ray_transfer_matrix(ns[-2], ns[-1])
                rt_mats[ii] = rt_next.dot(rt_mats[ii - 1])
            else:
                d = np.linalg.norm(self.surfaces[ii].paraxial_center - self.surfaces[ii - 1].paraxial_center)
                rt_surf = self.surfaces[ii - 1].get_ray_transfer_matrix(ns[ii - 1], ns[ii])
                rt_next = get_free_space_abcd(d, ns[ii]).dot(rt_surf)
                rt_mats[ii] = rt_next.dot(rt_mats[ii - 1])

        return rt_mats

    def get_cardinal_points(self,
                            wavelength: float,
                            initial_material: Material,
                            final_material: Material,
                            axis=None):
        """
        Get cardinal points of the system. These are the focal points, principal points, and nodal points.
        This function also returns the effective focal length

        :param wavelength:
        :param initial_material:
        :param final_material:
        :param axis: for non-radially symmetric objects, which axis
        :return fp1, fp2, pp1, pp2, np1, np2, efl1, efl2:
        """
        abcd_mat = self.get_ray_transfer_matrix(wavelength, initial_material, final_material)[-1]
        abcd_inv = self.reverse().get_ray_transfer_matrix(wavelength, final_material, initial_material)[-1]
        n_obj = initial_material.n(wavelength)
        n_img = final_material.n(wavelength)

        # ###############################################
        # find focal point to the right of the lens
        # ###############################################
        # if I left multiply my ray transfer matrix by free space matrix, then combined matrix has lens/focal form
        # for certain distance of propagation dx. Find this by setting A + d/n_img * C = 0
        d2 = -abcd_mat[0, 0] / abcd_mat[1, 0] * n_img

        # can also find the principal plane with the following construction
        # take income ray at (h1, n1*theta1 = 0). Consider the ray-transfer matrix which combines the optic and the
        # distance travelled dx to the focus
        # then extend the corresponding ray at (h2=0, n2*theta2) backwards until it reaches height h
        # can check this happens at position P2 = f2 + h1/theta2 (where theta2<0 here)
        # by construction the ray-transfer matrix above has A=0,
        # but in any case we have the relationship C*h1 = n2*theta2
        # or P2 = f2 + n2/C -> EFL2 = f2 - P2 = -n2/C

        # EFL = 1 / C, and this is not affect by multiplying the ray-transfer matrix by free-space propagation
        efl2 = -n_img / abcd_mat[1, 0]
        fp2 = self.surfaces[-1].paraxial_center + d2 * self.surfaces[-1].output_axis

        # find principal plane image space
        pp2 = fp2 - efl2 * self.surfaces[-1].output_axis

        # find nodal point in image space
        d2_nodal = (n_img - n_obj * abcd_inv[1, 1]) / abcd_inv[1, 0]
        np2 = self.surfaces[-1].paraxial_center + d2_nodal * self.surfaces[-1].output_axis

        # find focal point in object space
        d1 = -abcd_inv[0, 0] / abcd_inv[1, 0] * n_obj
        efl1 = -n_obj / abcd_inv[1, 0]
        fp1 = self.surfaces[0].paraxial_center - d1 * self.surfaces[0].input_axis

        # find principal plane in object space
        pp1 = fp1 + efl1 * self.surfaces[0].input_axis

        # find nodal point in object space
        d1_nodal = (n_obj - n_img * abcd_mat[1, 1]) / abcd_mat[1, 0]
        np1 = self.surfaces[0].paraxial_center - d1_nodal * self.surfaces[0].output_axis

        return fp1, fp2, pp1, pp2, np1, np2, efl1, efl2

    def auto_focus(self,
                   wavelength: float,
                   initial_material: Material,
                   final_material: Material,
                   mode: str = "ray-fan"):
        """
        Perform an autofocus operation. This function can handle rays which are
        initially collimated or initially diverging

        :param wavelength:
        :param initial_material:
        :param final_material:
        :param mode: "ray-fan", "collimated", "paraxial-focused", or "paraxial-collimated"
        :return f:
        """
        # todo: handle case where optical System extends past focus

        if mode == "ray-fan":
            # todo: maybe take sequence of rays with smaller and smaller angles...
            rays_focus = get_ray_fan([0, 0, 0], 1e-9, 3, wavelength)
            rays_focus = self.ray_trace(rays_focus, initial_material, final_material)
            focus = intersect_rays(rays_focus[-1, 1], rays_focus[-1, 2])[0]
        elif mode == "collimated":
            rays_focus = get_collimated_rays([0, 0, 0], 1e-9, 3, wavelength)
            rays_focus = self.ray_trace(rays_focus, initial_material, final_material)
            focus = intersect_rays(rays_focus[-1, 1], rays_focus[-1, 2])[0]
        elif mode == "paraxial-focused":
            f1, focus, _, _, _, _, _, _ = self.get_cardinal_points(wavelength, initial_material, final_material)
        elif mode == "paraxial-collimated":
            abcd = self.get_ray_transfer_matrix(wavelength,
                                                initial_material,
                                                final_material)[-1]
            # determine what free space propagation matrix we need such that initial ray (h, n*theta) -> (0, n'*theta')
            dx = -abcd[0, 0] / abcd[1, 0] * self.materials[-1].n(wavelength)

            focus = (self.surfaces[-1].paraxial_center[2] + dx * np.sign(self.surfaces[-1].input_axis[2]))
        else:
            raise ValueError(f"mode must be 'ray-fan', or 'collimated' 'paraxial-focused',"
                             f" or paraxial-collimated' but was '{mode:s}'")

        return focus

    def plot(self,
             ray_array: Optional[NDArray] = None,
             phi: float = 0,
             colors: Optional[list] = None,
             label: str = None,
             ax: Optional[Axes] = None,
             show_names: bool = True,
             fontsize: float = 16,
             **kwargs) -> (Figure, Axes):
        """
        Plot rays and optical surfaces

        :param ray_array: nsurfaces X nrays x 8
        :param phi: angle describing the azimuthal plane to plot. phi = 0 gives the meridional/tangential plane while
          phi = pi/2 gives the sagittal plane. # todo: not implemented for drawing the Surface projections
        :param colors: list of colors to plot rays
        :param label:
        :param ax: axis to plot results on. If None, a new figure will be generated
        :param show_names:
        :param fontsize:
        :param kwargs: passed through to figure, if it does not already exist
        :return fig_handle, axis:
        """
        # get axis to plot on
        if ax is None:
            figh = plt.figure(**kwargs)
            ax = plt.subplot(1, 1, 1)
        else:
            figh = ax.get_figure()

        # plot rays
        if ray_array is not None:
            # ray height in the desired azimuthal plane
            h_data = ray_array[:, :, 0] * np.cos(phi) + ray_array[:, :, 1] * np.sin(phi)

            if label is None:
                label = ""

            if colors is None:
                ax.plot(ray_array[:, :, 2], h_data, label=label)
            else:
                # ensure color argument is ok
                if len(colors) == 1 and not isinstance(colors, list):
                    colors = [colors] * ray_array.shape[1]

                if len(colors) != ray_array.shape[1]:
                    raise ValueError("len(colors) must equal ray_array.shape[1]")

                # plot each ray a different color
                for ii in range(ray_array.shape[1]):
                    if ii == 0:
                        ax.plot(ray_array[:, ii, 2], h_data[:, ii], color=colors[ii], label=label)
                    else:
                        ax.plot(ray_array[:, ii, 2], h_data[:, ii], color=colors[ii])

            ax.set_xlabel("z-position (mm)", fontsize=fontsize)
            ax.set_ylabel("height (mm)", fontsize=fontsize)

        ax.tick_params(axis='x', labelsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)

        # plot surfaces
        if self.surfaces is not None:
            for ii, s in enumerate(self.surfaces):
                s.draw(ax)
                if show_names:
                    if ii == 0 or self.surfaces_by_name[ii] != self.surfaces_by_name[ii - 1]:

                        ax.text(s.paraxial_center[2],  # x
                                s.paraxial_center[0] + 1.1 * s.aperture_rad,  # y
                                self.names[self.surfaces_by_name[ii]],  # s
                                horizontalalignment="center",
                                fontsize=fontsize
                                )

        return figh, ax


class Doublet(System):
    def __init__(self,
                 material_crown: Optional[Material] = None,
                 material_flint: Optional[Material] = None,
                 radius_crown: Optional[float] = None,
                 radius_flint: Optional[float] = None,
                 radius_interface: Optional[float] = None,
                 thickness_crown: Optional[float] = None,
                 thickness_flint: Optional[float] = None,
                 aperture_radius: float = 25.4,
                 input_collimated: bool = True,
                 names: str = ""):
        """
        Provide the radii of curvature assuming the lens is oriented with the crown side
        to the left (i.e. the proper orientation to focus a collimated beam oriented from the left)
        Positive curvature indicates a Surface appears convex when looking from the left. Most
        commonly the crown Surface will have a positive radius of curvatures while
        the intermediate Surface and flint Surface will have negative radii of curvature
        To create a lens oriented the other way, set input_collimated=True (but define the curvatures and
        other parameters as described above)

        :param material_crown:
        :param material_flint:
        :param radius_crown:
        :param radius_flint:
        :param radius_interface:
        :param thickness_crown:
        :param thickness_flint:
        :param aperture_radius:
        :param input_collimated: if True, the input goes into the crown Surface.
        :param names:
        """

        if input_collimated:
            m1 = material_crown
            m2 = material_flint
            if not np.isinf(radius_crown):
                s1 = SphericalSurface.get_on_axis(radius_crown, 0, aperture_radius)
            else:
                s1 = FlatSurface([0, 0, 0], [0, 0, 1], aperture_rad=aperture_radius)

            if not np.isinf(radius_interface):
                s2 = SphericalSurface.get_on_axis(radius_interface, thickness_crown, aperture_radius)
            else:
                s2 = FlatSurface([0, 0, thickness_crown], [0, 0, 1], aperture_rad=aperture_radius)

            if not np.isinf(radius_flint):
                s3 = SphericalSurface.get_on_axis(radius_flint, thickness_crown + thickness_flint, aperture_radius)
            else:
                s3 = FlatSurface([0, 0, thickness_crown + thickness_flint], [0, 0, 1], aperture_rad=aperture_radius)

        else:
            m1 = material_flint
            m2 = material_crown
            if not np.isinf(radius_flint):
                s1 = SphericalSurface.get_on_axis(-radius_flint,
                                                  0,
                                                  aperture_radius)
            else:
                s1 = FlatSurface([0, 0, 0],
                                 [0, 0, 1],
                                 aperture_rad=aperture_radius)

            if not np.isinf(radius_interface):
                s2 = SphericalSurface.get_on_axis(-radius_interface,
                                                  thickness_flint,
                                                  aperture_radius)
            else:
                s2 = FlatSurface([0, 0, thickness_flint],
                                 [0, 0, 1],
                                 aperture_rad=aperture_radius)

            if not np.isinf(radius_crown):
                s3 = SphericalSurface.get_on_axis(-radius_crown,
                                                  thickness_flint + thickness_crown,
                                                  aperture_radius)
            else:
                s3 = FlatSurface([0, 0, thickness_flint + thickness_crown],
                                 [0, 0, 1],
                                 aperture_rad=aperture_radius)

        self.radius_crown = float(radius_crown)
        self.radius_flint = float(radius_flint)
        self.radius_interface = float(radius_interface)
        self.thickness_crown = float(thickness_crown)
        self.thickness_flint = float(thickness_flint)
                
        super(Doublet, self).__init__([s1, s2, s3],
                                      [m1, m2],
                                      names=names,
                                      surfaces_by_name=None)


# ################################################
# optical surfaces
# ################################################
class Surface:
    """
    The geometry of the Surface should be defined by the center, input_axis, and output_axis
    """
    def __init__(self,
                 input_axis: NDArray,
                 output_axis: NDArray,
                 center: NDArray,
                 paraxial_center: NDArray,
                 aperture_rad: float):
        """

        :param input_axis:
        :param output_axis:
        :param center:
        :param paraxial_center:
        :param aperture_rad:
        """

        # input axes
        # n1_in, n2_in, n3_in are mutually orthogonal input unit vectors (think x, y, z).
        # n3_in is the input optical axis.
        # n1_in and n2_in are the x-like and y-like axes and the ABCD matrices are defined according to them
        self.input_axis = np.array(input_axis).squeeze().astype(float)

        # output axes, defined similar to n1_in, n2_in, n3_in
        # for input vectors infinitesimally away from the axis along n1_in, they will exit
        # displaced from the output optical axis along n1_out
        # this is mainly so ray-transfer matrices make sense for mirrors
        self.output_axis = np.array(output_axis).squeeze().astype(float)

        # center
        self.center = np.array(center).squeeze().astype(float)

        # paraxial center
        self.paraxial_center = np.array(paraxial_center).squeeze().astype(float)

        # aperture information
        self.aperture_rad = aperture_rad

    def get_normal(self, pts: array):
        """
        Get normal value at pts. Should be implemented so if pass a ray instead of a point, will
        take the (xo, yo, zo) part and ignore the rest

        :param pts:
        :return:
        """
        pass

    def get_intersect(self, rays: array, material: Material):
        """
        Find intersection points between rays and Surface for rays propagating through a given Material,
        and return resulting rays at intersection

        :param rays:
        :param material:
        :return:
        """
        pass

    def propagate(self,
                  ray_array: array,
                  material1: Material,
                  material2: Material):
        """
        propagate rays througn the Surface

        :param ray_array:
        :param material1:
        :param material2:
        :return:
        """
        pass

    def get_ray_transfer_matrix(self, n1: float, n2: float):
        """

        :param n1:
        :param n2:
        :return:
        """
        pass

    def solve_img_eqn(self,
                      s: NDArray,
                      n1: float,
                      n2: float):
        """
        solve imaging equation where s is the object distance and s' is the image distance

        s and s' use the same convention, where they are negative if to the "left" of the optic and positive
        to the right.

        :param s:
        :param n1:
        :param n2:
        :return sp_x:
        """
        mat = self.get_ray_transfer_matrix(n1, n2)
        # idea: form full ABCD matrix as free prop in n2 * mat * free prop in n1, then set B = 0
        with np.errstate(divide="ignore"):
            if np.abs(s) > 1e12:
                sp = np.atleast_1d(-n2 * mat[0, 0] / mat[1, 0])
            else:
                sp = np.atleast_1d(-n2 * (-mat[0, 0] * s / n1 + mat[0, 1]) / np.array(-mat[1, 0] * s / n1 + mat[1, 1]))

        return sp

    def is_pt_on_surface(self, pts: array):
        """
        Test if a point is on Surface

        :param pts:
        :return:
        """
        pass

    def draw(self, ax: Axes):
        """
        Draw Surface on matplotlib axis

        :param ax:
        :return:
        """
        pass


class RefractingSurface(Surface):
    def propagate(self,
                  ray_array: array,
                  material1: Material,
                  material2: Material) -> array:
        """
        Given a set of rays, propagate them to the Surface of this object and compute their refraction
        using Snell's law. Return the update ray array with these two new ray positions

        :param ray_array: nsurfaces x nrays x 8
        :param material1: Material on first side of Surface
        :param material2: Material on second side of Surface
        """
        xp = cp if cp is not None and isinstance(ray_array, cp.ndarray) else np
        input_axis = xp.asarray(self.input_axis)

        if ray_array.ndim == 1:
            ray_array = xp.expand_dims(ray_array, axis=(0, 1))
        if ray_array.ndim == 2:
            ray_array = xp.expand_dims(ray_array, axis=0)

        # get latest rays
        rays = ray_array[-1]
        rays_intersection = self.get_intersect(rays, material1)
        normals = self.get_normal(rays_intersection)

        # check ray was coming from the "front side"
        # i.e. the ray has to be coming from the correct 2*pi area of space
        ray_normals = rays[:, 3:6]
        cos_ray_input = xp.sum(ray_normals * xp.expand_dims(input_axis, axis=0), axis=1)
        with np.errstate(invalid="ignore"):
            not_incoming = cos_ray_input < 0
        rays_intersection[not_incoming] = xp.nan

        # ######################
        # do refraction
        # ######################
        # rays_refracted = refract(rays_intersection, normals, material1, material2)
        ds = rays_intersection[:, 3:6]
        wls = xp.expand_dims(rays_intersection[:, 7], axis=1)

        # basis for computation (normal, nb, nc)
        # nb orthogonal to normal and ray
        with np.errstate(invalid="ignore"):
            nb = xp.cross(ds, normals)
            nb = nb / xp.expand_dims(xp.linalg.norm(nb, axis=1), axis=1)
            nb[np.isnan(nb)] = 0

            nc = xp.cross(normals, nb)
            nc = nc / xp.expand_dims(xp.linalg.norm(nc, axis=1), axis=1)
            nc[xp.isnan(nc)] = 0

            # snell's law
            # the tangential component (i.e. nc direction) of k*n*ds is preserved across the interface
            mag_nc = material1.n(wls) / material2.n(wls) * xp.expand_dims(xp.sum(nc * ds, axis=1), axis=1)
            sign_na = xp.expand_dims(xp.sign(xp.sum(normals * ds, axis=1)), axis=1)
            # normalize outgoing ray direction. By construction nothing in nb direction
            ds_out = mag_nc * nc + sign_na * xp.sqrt(1 - mag_nc ** 2) * normals

            rays_refracted = xp.concatenate((rays_intersection[:, :3],
                                             ds_out,
                                             rays_intersection[:, 6:]), axis=1)
            rays_refracted[xp.isnan(ds_out[:, 0]), :3] = xp.nan

        # append refracted rays to full array
        ray_array = xp.concatenate((ray_array,
                                    xp.stack((rays_intersection,
                                              rays_refracted), axis=0)),
                                   axis=0)

        return ray_array


class ReflectingSurface(Surface):
    def propagate(self,
                  ray_array: array,
                  material1: Material,
                  material2: Optional[Material] = None) -> array:
        """
        Given a set of rays, propagate them to the Surface of this object and compute their reflection.
        Find new rays after reflecting off a Surface defined by a given normal using the law of reflection.
        Return the update ray array with these two new ray positions

        :param ray_array: nsurfaces x nrays x 8
        :param material1:
        :param material2:
        :return rays:
        """
        xp = cp if cp is not None and isinstance(ray_array, cp.ndarray) else np

        if ray_array.ndim == 1:
            ray_array = xp.expand_dims(ray_array, axis=(0, 1))
        if ray_array.ndim == 2:
            ray_array = xp.expand_dims(ray_array, axis=0)

        # get latest rays
        rays = ray_array[-1]
        rays_intersection = self.get_intersect(rays, material1)
        normals = self.get_normal(rays_intersection)

        # ############################
        # do reflection
        # ############################
        # basis for computation (normal, nb, nc)
        # nb orthogonal to normal and ray
        ds = rays_intersection[:, 3:6]
        with xp.errstate(invalid="ignore"):
            nb = xp.cross(ds, normals)
            nb = nb / xp.expand_dims(xp.linalg.norm(nb, axis=1), axis=1)
            nb[xp.isnan(nb)] = 0

            nc = xp.cross(normals, nb)
            nc = nc / xp.expand_dims(xp.linalg.norm(nc, axis=1), axis=1)
            nc[xp.isnan(nc)] = 0

        # law of reflection
        # the normal component (i.e. na direction) changes sign
        mag_na = -xp.expand_dims(xp.sum(normals * ds, axis=1), axis=1)
        mag_nc = xp.expand_dims(xp.sum(nc * ds, axis=1), axis=1)
        ds_out = mag_na * normals + mag_nc * nc

        rays_refracted = xp.concatenate((rays_intersection[:, :3],
                                   ds_out,
                                   rays_intersection[:, 6:]),
                                  axis=1)
        rays_refracted[xp.isnan(ds_out[:, 0]), :3] = xp.nan

        # append these rays to full array
        ray_array = xp.concatenate((ray_array,
                                    xp.stack((rays_intersection,
                                              rays_refracted),
                                             axis=0)),
                                   axis=0)

        return ray_array


class FlatSurface(RefractingSurface):
    """
    Surface is defined by
        [(x, y, z) - (cx, cy, cz)] \cdot normal = 0

    Where the normal should point along the intended direction of ray-travel
    """
    def __init__(self,
                 center,
                 normal,
                 aperture_rad: float):
        self.normal = np.array(normal).squeeze()
        super(FlatSurface, self).__init__(normal, normal, center, center, aperture_rad)

    def get_normal(self, pts: array):

        xp = cp if cp is not None and isinstance(pts, cp.ndarray) else np

        pts = xp.atleast_2d(pts)
        normal = xp.atleast_2d(self.normal)
        return xp.tile(normal, (pts.shape[0], 1))

    def get_intersect(self, rays: array, material: Material):
        rays_int, ts = propagate_ray2plane(rays,
                                           self.normal,
                                           self.center,
                                           material,
                                           exclude_backward_propagation=True)
        return rays_int

    def is_pt_on_surface(self, pts: array):

        xp = cp if cp is not None and isinstance(pts, cp.ndarray) else np

        pts = xp.atleast_2d(pts)
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
        xc, yc, zc = self.center
        nx, ny, nz = self.normal

        on_surface = xp.abs((x - xc) * nx + (y - yc) * ny + (z - zc) * nz) < 1e-12

        return on_surface

    def get_ray_transfer_matrix(self, n1=None, n2=None):
        mat = np.array([[1, 0],
                        [0, 1]])
        return mat

    def draw(self, ax: Axes):
        # take Y = 0 portion of Surface
        y_hat = np.array([0, 1, 0])
        normal_proj = self.normal - self.normal.dot(y_hat) * y_hat
        normal_proj = normal_proj / np.linalg.norm(normal_proj)

        # plane projected in XZ plane follows this direction
        dv = np.cross(normal_proj, y_hat)

        if not np.isinf(self.aperture_rad):
            ts = np.linspace(-self.aperture_rad, self.aperture_rad, 101)
        else:
            ts = np.array([0, 1])

        # construct points on-line using broadcasting
        pts = np.expand_dims(self.center, axis=0) + np.expand_dims(ts, axis=1) * np.expand_dims(dv, axis=0)

        if not np.isinf(self.aperture_rad):
            ax.plot(pts[:, 2], pts[:, 0], 'k')
        else:
            ax.axline(pts[0, (2, 0)], xy2=pts[1, (2, 0)], color='k')


class PlaneMirror(ReflectingSurface):
    """
    Surface is defined by
        [(x, y, z) - (cx, cy, cz)] \cdot normal = 0

    Where the normal should point along the intended direction of ray-travel
    """

    def __init__(self, center, normal, aperture_rad):
        self.normal = np.array(normal).squeeze()
        super(PlaneMirror, self).__init__(normal, normal, center, center, aperture_rad)

    def get_normal(self, pts: array):
        xp = cp if cp is not None and isinstance(pts, cp.ndarray) else np
        pts = xp.atleast_2d(pts)
        normal = xp.atleast_2d(self.normal)
        return xp.tile(normal, (pts.shape[0], 1))

    def get_intersect(self, rays: array, material: Material):
        rays_int, ts = propagate_ray2plane(rays, self.normal, self.center, material)
        # exclude rays which will not intersect plane (but would have intersected in the past)
        rays_int[ts < 0] = np.nan

        return rays_int

    def is_pt_on_surface(self, pts: array):
        xp = cp if cp is not None and isinstance(pts, cp.ndarray) else np
        pts = xp.atleast_2d(pts)
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
        xc, yc, zc = xp.asarray(self.center)
        nx, ny, nz = xp.asarray(self.normal)

        on_surface = xp.abs((x - xc) * nx + (y - yc) * ny + (z - zc) * nz) < 1e-12

        return on_surface

    def get_ray_transfer_matrix(self, n1: float, n2: float):
        mat = np.array([[1, 0], [0, -1]])
        return mat

    def draw(self, ax: Axes):
        # take Y = 0 portion of Surface
        y_hat = np.array([0, 1, 0])
        normal_proj = self.normal - self.normal.dot(y_hat) * y_hat
        normal_proj = normal_proj / np.linalg.norm(normal_proj)

        # plane projected in XZ plane follows this direction
        dv = np.cross(normal_proj, y_hat)

        ts = np.linspace(-self.aperture_rad, self.aperture_rad, 101)

        # construct line using broadcasting
        pts = np.expand_dims(self.center, axis=0) + np.expand_dims(ts, axis=1) * np.expand_dims(dv, axis=0)

        ax.plot(pts[:, 2], pts[:, 0], 'k')


class SphericalSurface(RefractingSurface):
    def __init__(self, radius, center, aperture_rad, input_axis=(0, 0, 1)):
        """
        Constructor method, but often it is more convenient to use get_on_axis() instead.

        :param radius:
        :param center: [cx, cy, cz]
        :param aperture_rad:
        :param input_axis:
        """
        self.radius = radius

        paraxial_center = np.array(center).squeeze() - self.radius * np.array(input_axis).squeeze()
        super(SphericalSurface, self).__init__(input_axis, input_axis, center, paraxial_center, aperture_rad)

    @classmethod
    def get_on_axis(cls,
                    radius: float,
                    surface_z_position: float,
                    aperture_rad: float):
        """
        Construct spherical Surface from position on-optical axis, instead of center
        Think of this as an alternate constructor

        :param radius:
        :param surface_z_position:
        :param aperture_rad:
        :return:
        """
        input_axis = (0, 0, 1)
        return cls(radius, [0, 0, surface_z_position + radius], aperture_rad, input_axis)

    def get_normal(self, pts: array) -> array:
        """
        Return the outward facing normal if self.aperture_radius > 0, otherwise the inward facing normal

        :param pts: each pt defined by row of matrix
        :return normals:
        """
        xp = cp if cp is not None and isinstance(pts, cp.ndarray) else np
        pts = xp.atleast_2d(pts)[:, :3]
        normals = (pts - xp.expand_dims(xp.asarray(self.center), axis=0)) / self.radius
        return normals

    def get_intersect(self, rays: array, material: Material) -> array:
        xp = cp if cp is not None and isinstance(rays, cp.ndarray) else np

        rays = xp.atleast_2d(rays)

        # ray parameterized by (xo, yo, zo) + t * (dx, dy, dz)
        # can show leads to quadratic equation for t with the following coefficients
        xo = rays[:, 0]
        yo = rays[:, 1]
        zo = rays[:, 2]
        dx = rays[:, 3]
        dy = rays[:, 4]
        dz = rays[:, 5]
        phase_o = rays[:, 6]
        wls = rays[:, 7]

        xc, yc, zc = self.center

        A = 1
        B = 2 * (dx * (xo - xc) + dy * (yo - yc) + dz * (zo - zc))
        C = (xo - xc)**2 + (yo - yc)**2 + (zo - zc)**2 - self.radius**2

        with np.errstate(invalid="ignore"):
            # we only want t > 0, since these are the forward points for the rays
            # and of the t > 0, we want the smallest t
            ts = xp.stack((0.5 * (-B + xp.sqrt(B**2 - 4 * A * C)),
                           0.5 * (-B - xp.sqrt(B**2 - 4 * A * C))), axis=1)
            ts[ts < 0] = np.inf

        t_sol = xp.min(ts, axis=1)
        t_sol[t_sol == np.inf] = np.nan

        pts = xp.stack((xo, yo, zo), axis=1) + xp.stack((dx, dy, dz), axis=1) * xp.expand_dims(t_sol, axis=1)
        phase_shift = xp.linalg.norm(pts - xp.stack((xo, yo, zo), axis=1), axis=1) * 2 * np.pi / wls * material.n(wls)

        rays_int = xp.concatenate((pts, xp.stack((dx, dy, dz, phase_o + phase_shift, wls), axis=1)), axis=1)

        return rays_int

    def is_pt_on_surface(self, pts: array) -> array:
        """
        Check if point is on sphere surfaces

        :param pts:
        :return:
        """
        xp = cp if cp is not None and isinstance(pts, cp.ndarray) else np
        pts = xp.atleast_2d(pts)
        diff = (pts[:, 0] - self.center[0]) ** 2 + (pts[:, 1] - self.center[1]) ** 2 + (pts[:, 2] - self.center[2]) ** 2
        on_surface = xp.abs(diff - self.radius**2) < 1e-12
        return on_surface

    def get_ray_transfer_matrix(self, n1: float, n2: float) -> NDArray:
        # test if we are going from "inside the sphere" to "outside the sphere" i.e. ray is striking the concave side
        # or the other way
        pc_to_c = self.center - self.paraxial_center
        sgn = np.sign(np.dot(pc_to_c, self.input_axis))

        with np.errstate(divide="ignore"):
            f = sgn * np.abs(self.radius) / np.array(n2 - n1)

        mat = np.array([[1,    0], [-1/f, 1]])
        return mat

    def draw(self, ax: Axes):
        # todo: modify to allow arbitrary input axis
        theta_max = np.arcsin(self.aperture_rad / np.abs(self.radius))
        thetas = np.linspace(-theta_max, theta_max, 101)
        pts_z = self.center[2] - self.radius * np.cos(thetas)
        pts_x = self.center[0] - self.radius * np.sin(thetas)
        ax.plot(pts_z, pts_x, 'k')


class PerfectLens(RefractingSurface):
    def __init__(self,
                 focal_len: float,
                 center: NDArray,
                 normal: NDArray,
                 alpha: float):
        """
        This lens has no length. The center position defines the principal planes.
        The normal should point the same direction rays propagate through the System
        The focal point is focal_len away from the Surface

        :param focal_len:
        :param center:
        :param normal:
        :param alpha: maximum angle
        """
        self.focal_len = focal_len
        self.alpha = alpha
        self.normal = np.array(normal).squeeze()
        aperture_rad = focal_len * np.sin(self.alpha)  # only correct up to factor of n1
        super(PerfectLens, self).__init__(normal, normal, center, center, aperture_rad)

    def get_intersect(self, rays: array, material: Material) -> array:
        rays_int, ts = propagate_ray2plane(rays, self.normal, self.center, material)
        with np.errstate(invalid="ignore"):
            rays_int[ts < 0] = np.nan
        return rays_int

    def is_pt_on_surface(self, pts: array) -> array:

        xp = cp if cp is not None and isinstance(pts, cp.ndarray) else np

        pts = xp.atleast_2d(pts)
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
        xc, yc, zc = self.center
        nx, ny, nz = self.normal

        on_surface = xp.abs((x - xc) * nx + (y - yc) * ny + (z - zc) * nz) < 1e-12

        return on_surface

    def propagate(self,
                  rays: array,
                  material1: Material,
                  material2: Material) -> array:
        """
        Given a set of rays, propagate them to the Surface of this object and compute their refraction.
        Return the update ray array with these two new ray positions

        Construction: consider lens as a plane Surface distance f from FFP and BFP. Consider rays
        in the FFP. We know that an on axis ray at angle theta. We know this becomes an on-axis ray
        at distance n*focal len * sin(theta) from the optical axis.

        Suppose we have the Fourier shift theorem perfectly satisfied, then we can use this to infer the angles
        of the other rays, recalling that in the BFP the spatial frequency f = -xp / fl / lambda
        e.g. for 1D case sin(theta_p) = xo/fl

        So the full mapping from FFP to BFP is
        (h, sin(theta) ) -> (n1 * fl * sin(theta), -h / fl / n2)
        And the front and back focal planes are located distance n1*f before and n2*f after the lens respectively

        We can see that cascading two of these lenses together ensures the Abbe sine condition is satisfied

        Note: there will be a positional discontinuity at the Surface of the lens. For example, given a beam parallel
        to the optical axis incident at height h, let's compute the height after the lens, determined by the angle
        which the beam focuses. To see this, note
        We know theta = sin^(-1)(x / fl) and the
        new height = f*tan(theta) = f * h / sqrt(fl^2-h^2) ~ x + 0.5 * x^3 / fl^2 + ...
        so the heights only match to first order

        To calculate the phase change introduced by the lens imagine the following situation: take an arbitrary
        with angle theta to the axis and height h in the front focal plane of the lens. Suppose it is part of
        a plane wave with many parallel rays. Since this is a perfect lens, these rays come to a common focus
        (with the same phase) in the back focal plane of the lens at the position we computed above. So in the
        back focal plane all of these rays must have the same phase. For convenience, we suppose the optical
        path length between the front and back focal planes is (n1**2 + n2**2) * f. This follows because in
        the immersion medium the focal planes are distance n1*f and n2*f away from the lens.

        If we suppose the parallel ray passing through the front focal point has OPL at this point, then in the
        focal plane our ray has extra phase n1*h_1*sin(theta_1). The rays travel the same distance to the lens.
        Directly after the lens, these two rays have height difference n2 * f * tan(theta_2). The path length
        difference is n2 * f * (1 / cos(theta_2) - 1). Since the total phase difference is 0, we must have
        lens_phase(h1, theta_1) = -n2**2 * f * (1 / cos(theta_2) - 1) - n1 * h1*sin(theta_1)
                                = -n2**2 * f * [1 / sqrt(1 - h1^2 / f^2 / n_2^2) - 1] - n1 * h1*sin(theta_1)

        In practice, this expression is not implemented directly. Rather, equal phase is imposed at the focus

        When considering the vectorial model, we instead talk about the ray position vector r1, the ray position
        vector relative to the focal point r1', the ray unit vector s1, and the unit vector formed by projecting
        the optical axis direction n from s1 call it s1' = (s1 - n\cdot s1) / |s_1 - n\cdot s1|. In terms
        of these quantities,
        r2' = n1 * f * (s1' \cdot s1) * s1'
        s2' = - |r1'| / (n2 * f) * (r1' / |r1'|)

        :param rays: nsurfaces x nrays x 8
        :param material1: Material on first side of Surface
        :param material2: Material on second side of Surface
        :return rays_out:
        """

        xp = cp if cp is not None and isinstance(rays, cp.ndarray) else np
        center = xp.asarray(self.center)
        normal = xp.asarray(self.normal)

        if rays.ndim == 1:
            rays = xp.expand_dims(rays, axis=(0, 1))
        elif rays.ndim == 2:
            rays = xp.expand_dims(rays, axis=0)

        wls = rays[-1, :, 7]

        # #####################################
        # get the three surfaces we will need in our calculation: front focal plane (i.e. before the lens),
        # lens Surface, back focal plane (i.e. after the lens)
        # #####################################
        front_focal_pts = (xp.expand_dims(center, axis=0) -
                           xp.expand_dims(normal, axis=0) * self.focal_len *
                           xp.expand_dims(material1.n(wls), axis=1))
        back_focal_pts = (xp.expand_dims(center, axis=0) +
                          xp.expand_dims(normal, axis=0) * self.focal_len *
                          xp.expand_dims(material2.n(wls), axis=1))

        # #####################################
        # find position rays intersect the object plane (front focal plane)
        # if rays are already in front of the object plane, propagate them backwards to reach it
        # #####################################
        rays_ffp, _ = propagate_ray2plane(rays[-1],
                                          normal,
                                          front_focal_pts,
                                          material1,
                                          exclude_backward_propagation=False)

        # #####################################
        # compute geometric data (height and angle) of rays in ffp relative to the lens axis
        # #####################################

        # get unit vectors pointing along input rays
        s1 = rays_ffp[:, 3:6]
        # n0 are the unit vectors pointing along rays after projecting out the portion along the lens normal
        ray_normal_dot = xp.sum(s1 * xp.expand_dims(normal, axis=0), axis=1)
        s1_perp_uvec = s1 - xp.expand_dims(ray_normal_dot, axis=1) * xp.expand_dims(normal, axis=0)

        # only normalize non-zero rays
        with np.errstate(invalid="ignore"):
            # to_normalize = np.sum(np.abs(n0), axis=1) > 1e-13
            # n0[to_normalize] = n0[to_normalize] / np.expand_dims(np.linalg.norm(n0[to_normalize], axis=1), axis=1)
            s1_perp_norm = xp.linalg.norm(s1_perp_uvec, axis=1)
            to_normalize = s1_perp_norm > 1e-12
            s1_perp_uvec[to_normalize] = s1_perp_uvec[to_normalize] / xp.expand_dims(s1_perp_norm[to_normalize], axis=1)

        # compute the vectorial "height" of rays above optical axis
        # r1_vec is the vector in the FFP pointing to the position of the ray
        # i.e. the ray position after projecting out the optical axis direction
        r1_vec = rays_ffp[:, 0:3] - front_focal_pts
        r1_norm = xp.linalg.norm(r1_vec, axis=1)

        # get unit vector
        to_normalize = r1_norm != 0
        r1_uvec = xp.array(r1_vec, copy=True)
        r1_uvec[to_normalize] = r1_uvec[to_normalize] / xp.expand_dims(xp.linalg.norm(r1_uvec[to_normalize],
                                                                                      axis=1),
                                                                       axis=1)

        # sine of angle between incoming ray and the optical axis
        sin_t1 = xp.sum(s1_perp_uvec * s1, axis=1)

        # #####################################
        # construct rays as they appear in the BFP (i.e. after the lens)
        # #####################################
        rays_bfp = xp.zeros(rays_ffp.shape)

        # keep same wavelengths
        rays_bfp[:, 7] = wls

        # compute ray positions in BFP. These depend only on the input direction and are found by
        # vectorial position = h * n0 where h = n*fl*sin(theta_1)
        h2 = xp.expand_dims(material1.n(wls), axis=1) * self.focal_len * xp.expand_dims(sin_t1, axis=1) * s1_perp_uvec
        rays_bfp[:, :3] = h2 + xp.expand_dims(back_focal_pts, axis=0)

        # output angles
        # get unit vector for input positions, r0
        with np.errstate(invalid="ignore"):
            sin_t2 = -r1_norm / self.focal_len / material2.n(wls)
            cos_t2 = xp.sqrt(1 - sin_t2**2)
            rays_bfp[:, 3:6] = xp.expand_dims(sin_t2, axis=1) * r1_uvec + \
                               xp.expand_dims(cos_t2, axis=1) * xp.expand_dims(normal, axis=0)

        # #####################################
        # exclude points which are outside the NA of the lens
        # #####################################
        with np.errstate(invalid="ignore"):
            input_angle_too_steep = xp.abs(sin_t1) > xp.sin(self.alpha)
            output_angle_too_steep = xp.abs(sin_t2) > xp.sin(self.alpha)
            rays_bfp[xp.logical_or(input_angle_too_steep, output_angle_too_steep)] = xp.nan

        # #####################################
        # set phase at BfP. There are three contributions:
        # (1) the initial phase
        # (2) the phase the lens imparts. This can be calculated by realizing the final phases must be equal
        # for a fan of parallel rays (i.e. a plane wave). Since in the initial plane these rays have phases
        # k*n1*h1*sin_t1, we must subtract this phase here. The rest of the lens phase shift is taken care of
        # by enforcing the phases initially being equal in the BFP
        # note that h1 is not the signed height. To get the signed height, we must check if n0 and r0 point
        # in the same direction (positive height) or opposite directions (negative height)
        # (3) the phase from propagating distance n1*f + n2*f which is (k*n1)*n1*f + (k*n2)*n2*f
        # #####################################
        plane_wave_phase = xp.sum(r1_vec * s1, axis=1)

        rays_bfp[:, 6] = rays_ffp[:, 6] - \
                         2 * np.pi / wls * material1.n(wls) * plane_wave_phase + \
                         2 * np.pi / wls * (material1.n(wls)**2 * self.focal_len + material2.n(wls)**2 * self.focal_len)

        # #####################################
        # propagate rays from bfp backwards to lens position
        # note: from this direction don't care about aperture
        # #####################################
        rays_after_lens, _ = propagate_ray2plane(rays_bfp,
                                                 normal,
                                                 center,
                                                 material2,
                                                 exclude_backward_propagation=False)

        # also need position that the rays would normally intersect lens position
        rays_before_lens, _ = propagate_ray2plane(rays[-1],
                                                  normal,
                                                  center,
                                                  material1)

        # #####################################
        # output ray array is all the rays that were passed in and two new Surface
        # (1) right before the lens and (2) right after
        # #####################################
        rays_out = xp.concatenate((rays, xp.stack((rays_before_lens, rays_after_lens), axis=0)), axis=0)

        return rays_out

    def get_ray_transfer_matrix(self, n1: float, n2: float) -> NDArray:
        mat = np.array([[1, 0], [-1/self.focal_len, 1]])
        return mat

    def draw(self, ax: Axes):
        # take Y = 0 portion of Surface
        y_hat = np.array([0, 1, 0])
        normal_proj = self.normal - self.normal.dot(y_hat) * y_hat
        normal_proj = normal_proj / np.linalg.norm(normal_proj)

        # plane projected in XZ plane follows this direction
        dv = np.cross(normal_proj, y_hat)

        ts = np.linspace(-self.aperture_rad, self.aperture_rad, 101)

        # construct line using broadcasting
        pts = np.expand_dims(self.center, axis=0) + np.expand_dims(ts, axis=1) * np.expand_dims(dv, axis=0)

        ax.plot(pts[:, 2], pts[:, 0], 'k')
