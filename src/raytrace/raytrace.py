"""
ray represented by eight parameters
(xo, yo, zo, dx, dy, dz, phase, wavelength)
where (xo, yo, zo) is a point the ray passes through, and (dx, dy, dz) is a
unit vector specifying its direction

The ray travels along the curve C(t) = (xo, yo, zo) + t * (dx, dy, dz)

A paraxial ray has a different representation
(hx, n*ux, hy, n*uy) defined relative to some point and axis.
where (xo, yo, zo) and (dx, dy, dz) define the paraxial axis.

col([hx, n*ux, hy, n*uy, 1]) = [[A, B, 0, 0, K],
                                [C, D, 0, 0, L],
                                [0, 0, E, F, M],
                                [0, 0, G, H, N],
                                [0, 0, 0, 0, 1]] * col([hx1, n1*ux1, hy2, n2*uy2, 1]
here K, L account for any shift between the definition fo hx, n*ux and the natural center points.

z points to the right, along the optical axis. x points upwards, and y points out of the plane,
ensuring the coordinate System is right-handed
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


# def compute_third_order_seidel(surfaces: list,
#                                materials: list[Material],
#                                wavelength: float):
#     """
#
#     :param surfaces:
#     :param materials:
#     :param wavelength:
#     :return aberrations_3rd:
#     """
#     # first check where the aperture stop is
#     is_ap_stop = [s.is_aperture_stop for s in surfaces]
#
#     if not np.any(is_ap_stop):
#         raise ValueError("none of the surfaces were labelled as the aperture stop")
#     if np.sum(is_ap_stop) > 1:
#         raise ValueError("more than one of the surfaces was labelled as the aperture stop")
#
#     istop = int(np.where(is_ap_stop)[0])
#
#     # find entrance pupil
#     surfaces_before_stop = surfaces[:istop + 1]
#     materials_before_stop = materials[:istop + 2]
#
#     surfaces_before_stop, materials_before_stop = get_reversed_system(surfaces_before_stop, materials_before_stop)
#
#     # to avoid refracting through the stop again, modify the index of refraction
#     materials_before_stop[0] = materials_before_stop[1]
#
#     s_temp, _ = auto_focus(surfaces_before_stop, materials_before_stop, wavelength, mode="paraxial-focused")
#     # correct entrance pupil directions
#     entrance_pupil, _ = get_reversed_system([s_temp[-1]], [1, 1])
#     entrance_pupil = entrance_pupil[0]
#
#     # find exit pupil
#     # surfaces_after_stop = surfaces[istop:]
#     # ns_after_stop = materials[istop - 1:]
#     #
#     # s_temp, _ = auto_focus(surfaces_after_stop, ns_after_stop, mode="paraxial")
#     # exit_pupil = s_temp[-1]
#
#     # needed quantities
#     nsurfaces = len(surfaces)
#     aberrations_3rd = np.zeros((nsurfaces, 5))
#     s = np.zeros(nsurfaces)  # object points per Surface
#     sp = np.zeros(nsurfaces)  # image points per Surface
#     h = np.zeros(nsurfaces)  # h1 = s1 / (t1 - s1) where t = distance from entrance pupil to Surface vertex
#     H = np.zeros(nsurfaces)  # H1 = t1 / no
#     t = np.zeros(nsurfaces)
#     tp = np.zeros(nsurfaces)
#     d = np.zeros(nsurfaces - 1)
#
#     # initialize values for first Surface
#     s[0] = surfaces[0].paraxial_center[2]
#     sp[0] = surfaces[0].solve_img_eqn(s[0], materials[0], materials[1])
#     t[0] = entrance_pupil.paraxial_center[2] - surfaces[0].paraxial_center[2]
#     tp[0] = surfaces[0].solve_img_eqn(t[0], materials[0], materials[1])
#     h[0] = s[0] / (t[0] - s[0])
#     H[0] = t[0] / materials[0]
#
#     # compute subsequent values of image positions and pupil positions
#     for ii in range(1, len(h)):
#         # d[ii] is the distance between ii and ii + 1 surfaces
#         d[ii - 1] = surfaces[ii].paraxial_center[2] - surfaces[ii - 1].paraxial_center[2]
#
#         # new object is previous image relative to new surfaces
#         s[ii] = sp[ii - 1] - d[ii - 1]
#         sp[ii] = surfaces[ii].solve_img_eqn(s[ii], materials[ii], materials[ii + 1])
#         # new pupil is previous image relative to new surfaces
#         t[ii] = tp[ii - 1] - d[ii - 1]
#         tp[ii] = surfaces[ii].solve_img_eqn(t[ii], materials[ii], materials[ii + 1])
#
#         # see Born and Wolf chapter 5.5 eq's (9) and (16)
#         # h[ii] = h[ii - 1] * s[ii] / sp[ii - 1] # recursion has problem if one is zero...
#         h[ii] = s[ii] / (t[ii] - s[ii])
#         # H[ii] = H[ii - 1] * t[ii] / tp[ii - 1]
#         H[ii] = t[ii] / materials[ii]
#
#     # solve for aberrations
#     for ii in range(nsurfaces):
#         aberrations_3rd[ii] = surfaces[ii].get_seidel_third_order_fns(
#                                             materials[ii], materials[ii + 1],
#                                             s[ii], sp[ii], t[ii], tp[ii], h[ii], H[ii])
#
#     return aberrations_3rd


# propagation and refraction
def get_free_space_abcd(d: float, n: float = 1.):
    """
    Compute the ray-transfer (ABCD) matrix for free space beam propagation

    :param d: distance beam propagated
    :param n: index of refraction
    :return mat:
    """
    mat = np.array([[1, d/n, 0, 0,   0],
                    [0, 1,   0, 0,   0],
                    [0, 0,   1, d/n, 0],
                    [0, 0,   0, 1,   0],
                    [0, 0,   0, 0,   1]])
    return mat


# tools for creating ray fans, manipulating rays, etc.
def get_ray_fan(pt: NDArray,
                theta_max: float,
                n_thetas: int,
                wavelengths: float,
                nphis: int = 1,
                center_ray=(0, 0, 1)):
    """
    Get fan of rays emanating from pt

    :param pt: [cx, cy, cz]
    :param theta_max: maximum angle in radians
    :param n_thetas: number of rays at different angles on axis
    :param wavelengths:
    :param nphis: number of points of rotation about the optical axis. If nphis = 1, all rays will be in the plane
    :param center_ray:
    """

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


def intersect_rays(ray1: NDArray, ray2: NDArray) -> NDArray:
    """
    Find intersection point between two rays, assuming free space propagation

    if either s or t is negative then these rays previously intersected

    :param ray1:
    :param ray2:
    :return intersection_pt:
    """
    ray1 = np.atleast_2d(ray1)
    ray2 = np.atleast_2d(ray2)

    if len(ray1) == 1 and len(ray2) > 1:
        ray1 = np.tile(ray1, (len(ray2), 1))

    if len(ray2) == 1 and len(ray1) > 1:
        ray2 = np.tile(ray2, (len(ray1), 1))

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
    s = np.zeros(len(ray1)) * np.nan
    with np.errstate(invalid="ignore"):
        use_xz = dx2 * dz1 - dz2 * dx1 != 0
        use_xy = np.logical_and(np.logical_not(use_xz),
                                dx2 * dy1 - dy2 * dx1)
        use_yz = np.logical_and.reduce((np.logical_not(use_xz),
                                        np.logical_not(use_xy),
                                        dz2 * dy1 - dy2 * dz1))

        s[use_xz] = (((z2 - z1) * dx1 - (x2 - x1) * dz1) / (dx2 * dz1 - dz2 * dx1))[use_xz]
        s[use_xy] = (((y2 - y1) * dx1 - (x2 - x1) * dy1) / (dx2 * dy1 - dy2 * dx1))[use_xy]
        s[use_yz] = (((y2 - y1) * dz1 - (z2 - z1) * dy1) / (dz2 * dy1 - dy2 * dz1))[use_yz]
        # otherwise, d1 \cross d2 = 0, so rays are parallel and leave as NaN

    # determine distance along ray2, but avoid undefined expressions
    t = np.zeros(len(ray1)) * np.nan
    with np.errstate(all="ignore"):
        use_z = dz1 != 0
        use_y = np.logical_and(np.logical_not(use_z), dy1 != 0)
        # otherwise dx1 is guaranteed to be != 0 since dr1 is a unit vector
        use_x = np.logical_not(np.logical_or(use_z, use_y))
        t[use_z] = ((z2 + s * dz2 - z1) / dz1)[use_z]
        t[use_y] = ((y2 + s * dy2 - y1) / dy1)[use_y]
        t[use_x] = ((x2 + s * dx2 - x1) / dx1)[use_x]

    # but to verify the solution, must check intersection points are actually equal
    intersect1 = np.stack((x1, y1, z1), axis=1) + np.expand_dims(t, axis=1) * np.stack((dx1, dy1, dz1), axis=1)
    intersect2 = np.stack((x2, y2, z2), axis=1) + np.expand_dims(s, axis=1) * np.stack((dx2, dy2, dz2), axis=1)

    with np.errstate(invalid="ignore"):
        not_sol = np.max(np.abs(intersect1 - intersect2), axis=1) > 1e-12
        intersect1[not_sol] = np.nan

    return intersect1


def propagate_ray2plane(rays: NDArray,
                        normal: NDArray,
                        center: NDArray,
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
    rays = np.atleast_2d(np.array(rays, copy=True))

    normal = np.array(normal).squeeze()
    if normal.ndim == 1:
        normal = np.expand_dims(normal, axis=0)

    center = np.array(center).squeeze()
    if center.ndim == 1:
        center = np.expand_dims(center, axis=0)

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
        prop_direction = np.ones(rays.shape[0], dtype=int)
        prop_direction[ts < 0] = -1

    # find intersection points
    prop_dist_vect = np.stack((dx, dy, dz), axis=1) * np.expand_dims(ts, axis=1)
    pts = np.stack((xo, yo, zo), axis=1) + prop_dist_vect
    phase_shift = np.linalg.norm(prop_dist_vect, axis=1) * prop_direction * 2 * np.pi / wls * material.n(wls)

    # assemble output rays
    rays_out = np.concatenate((pts, np.stack((dx, dy, dz, phase_o + phase_shift, wls), axis=1)), axis=1)

    # replace back propagating rays with Nans if desired
    if exclude_backward_propagation:
        rays_out[prop_direction == -1, :] = np.nan

    return rays_out, ts


def ray_angle_about_axis(rays: NDArray, reference_axis: NDArray) -> (NDArray, NDArray):
    """
    Given a set of rays, compute their angles relative to a given axis, and compute the orthogonal direction to the

    axis which the ray travels in
    :param rays:
    :param reference_axis:
    :return angles, na:
    """
    rays = np.atleast_2d(rays)

    cosines = np.sum(rays[:, 3:6] * np.expand_dims(reference_axis, axis=0), axis=1)
    angles = np.arccos(cosines)
    na = rays[:, 3:6] - np.expand_dims(cosines, axis=1) * np.expand_dims(reference_axis, axis=0)
    na = na / np.expand_dims(np.linalg.norm(na, axis=1), axis=1)

    return angles, na


def dist_pt2plane(pts: NDArray,
                  normal: NDArray,
                  center: NDArray) -> (NDArray, NDArray):
    """
    Calculate minimum distance between points and a plane defined by normal and center

    :param pts:
    :param normal:
    :param center:
    :return dists, nearest_pts:
    """
    pts = np.atleast_2d(pts)
    npts = pts.shape[0]

    rays = np.concatenate((pts, np.tile(normal, (npts, 1)), np.zeros((npts, 2))), axis=1)
    rays_int, _ = propagate_ray2plane(rays, normal, center, Vacuum())

    dists = np.linalg.norm(rays_int[:, :3] - pts, axis=1)
    nearest_pts = rays_int[:, :3]

    return dists, nearest_pts


# ################################################
# collections of optical elements
# ################################################
class System:
    """
    Collection of optical surfaces
    """

    def __init__(self,
                 surfaces: list,
                 materials: list[Material],
                 names: list[str] = None,
                 surfaces_by_name=None):
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

        # todo: also need a way to carry names/descriptions of lenses around and show on plot

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
            new_surfaces_by_name = other.surfaces_by_name
            new_names = other.names
        elif isinstance(other, Surface):
            new_surfaces = [copy.deepcopy(other)]
            new_materials = []
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

        return System(self.surfaces + new_surfaces,
                      self.materials + [material] + new_materials,
                      names=self.names + new_names,
                      surfaces_by_name=surfaces_by_name)

    def find_paraxial_collimated_distance(self,
                                          other,
                                          wavelength: float,
                                          initial_material: Material,
                                          intermediate_material: Material,
                                          final_material: Material) -> (float, float):
        """
        Given two sets of surfaces (e.g. two lenses) determine the distance which should be inserted between them
        to give a System which converts collimated rays to collimated rays

        :param other:
        :param wavelength:
        :param initial_material:
        :param intermediate_material:
        :param final_material:
        :return dx, dy:
        """
        mat1 = self.get_ray_transfer_matrix(wavelength, initial_material, intermediate_material)
        mat2 = other.get_ray_transfer_matrix(wavelength, intermediate_material, final_material)

        dx = -(mat1[0, 0] / mat1[1, 0] + mat2[1, 1] / mat2[1, 0]) * intermediate_material.n(wavelength)
        dy = -(mat1[2, 2] / mat1[3, 2] + mat2[3, 3] / mat2[3, 2]) * intermediate_material.n(wavelength)

        return dx, dy

    def ray_trace(self,
                  rays: NDArray,
                  initial_material: Material,
                  final_material: Material) -> NDArray:
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

        rays = np.array(rays)

        for ii in range(len(self.surfaces)):
            rays = self.surfaces[ii].propagate(rays, materials[ii], materials[ii + 1])

        return rays

    def get_ray_transfer_matrix(self,
                                wavelength: float,
                                initial_material: Material,
                                final_material: Material,
                                axis=None):
        """
        Generate the ray transfer (ABCD) matrix for an optical System
        Assume that the optical System starts at the provided initial distance before the first Surface

        :param wavelength:
        :param initial_material:
        :param final_material:
        :param axis: axis which should be a direction orthogonal to the main beam direction
        :return abcd_matrix:
        """
        surfaces = self.surfaces
        materials = [initial_material] + self.materials + [final_material]

        indices_of_refraction = [m.n(wavelength) for m in materials]

        # propagate initial distance
        ray_xfer_mat = get_free_space_abcd(0, n=indices_of_refraction[0])
        for ii in range(len(surfaces)):
            # apply ray transfer matrix for current Surface
            abcd_temp = surfaces[ii].get_ray_transfer_matrix(indices_of_refraction[ii], indices_of_refraction[ii + 1])
            ray_xfer_mat = abcd_temp.dot(ray_xfer_mat)

            # apply ray transfer matrix propagating between surfaces
            if ii < (len(surfaces) - 1):
                dist = np.linalg.norm(surfaces[ii].paraxial_center - surfaces[ii + 1].paraxial_center)
                abcd_prop = get_free_space_abcd(dist, n=indices_of_refraction[ii + 1])
                ray_xfer_mat = abcd_prop.dot(ray_xfer_mat)

        return ray_xfer_mat

    def get_cardinal_points(self,
                            wavelength: float,
                            initial_material: Material,
                            final_material: Material):
        """

        :param wavelength:
        :param initial_material:
        :param final_material:
        :return fp1, fp2, pp1, pp2, np1, np2, efl1, efl2:
        """
        # todo: add nodal points computation
        # ###############################################
        # find focal point to the right of the lens
        # ###############################################
        abcd_mat = self.get_ray_transfer_matrix(wavelength, initial_material, final_material)
        n = final_material.n(wavelength)

        # if I left multiply my ray transfer matrix by free space matrix, then combined matrix has lens/focal form
        # for certain distance of propagation dx. Find this by setting A + d/n * C = 0
        d2x = -abcd_mat[0, 0] / abcd_mat[1, 0] * n
        d2y = -abcd_mat[2, 2] / abcd_mat[3, 2] * n

        abcd_mat_x = get_free_space_abcd(d2x, n).dot(abcd_mat)
        abcd_mat_y = get_free_space_abcd(d2y, n).dot(abcd_mat)

        # can also find the principal plane with the following construction
        # take income ray at (h1, n1*theta1 = 0). Consider the ray-transfer matrix which combines the optic and the
        # distance travelled dx to the focus
        # then extend the corresponding ray at (h2=0, n2*theta2) backwards until it reaches height h
        # can check this happens at position P2 = f2 + h1/theta2 (where theta2<0 here)
        # by construction the ray-transfer matrix above has A=0,
        # but in any case we have the relationship C*h1 = n2*theta2
        # or P2 = f2 + n2/C -> EFL2 = f2 - P2 = -n2/C

        # EFL = 1 / C, and this is not affect by right or left-multiplying
        # the ray-transfer matrix by free space propagation
        efl2_x = -n / abcd_mat_x[1, 0]
        efl2_y = -n / abcd_mat_y[3, 2]

        d2 = 0.5 * (d2x + d2y)
        efl2 = 0.5 * (efl2_x + efl2_y)

        # todo: is this still right in presence of RI?
        fp2 = self.surfaces[-1].paraxial_center + d2 * self.surfaces[-1].output_axis

        # find principal plane to the right
        pp2 = fp2 - efl2 * self.surfaces[-1].output_axis
        np2 = None

        # ##################################
        # find focal point to the left of the lens
        # ##################################
        abcd_inv = self.reverse().get_ray_transfer_matrix(wavelength, final_material, initial_material)

        d1x = -abcd_inv[0, 0] / abcd_inv[1, 0] * n
        d1y = -abcd_inv[2, 2] / abcd_inv[3, 2] * n

        abcd_mat_x_right = get_free_space_abcd(d1x, n).dot(abcd_inv)
        abcd_mat_y_right = get_free_space_abcd(d1y, n).dot(abcd_inv)
        efl1_x = -n / abcd_mat_x_right[1, 0]
        efl1_y = -n / abcd_mat_y_right[3, 2]

        d1 = 0.5 * (d1x + d1y)
        efl1 = 0.5 * (efl1_x + efl1_y)
        fp1 = self.surfaces[0].paraxial_center - d1 * self.surfaces[0].input_axis

        # find principal plane to the left of the lens
        pp1 = fp1 + efl1 * self.surfaces[0].input_axis
        np1 = None

        return fp1, fp2, pp1, pp2, np1, np2, efl1, efl2

    def auto_focus(self,
                   wavelength: float,
                   initial_material: Material,
                   final_material: Material,
                   mode: str = "ray-fan"):
        """
        Perform an auto-focus operation. This function can handle rays which are
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
                                                final_material)
            # determine what free space propagation matrix we need such that initial ray (h, n*theta) -> (0, n'*theta')
            dx = -abcd[0, 0] / abcd[1, 0] * self.materials[-1].n(wavelength)
            dy = -abcd[2, 2] / abcd[3, 2] * self.materials[-1].n(wavelength)

            if np.abs(dx - dy) >= 1e-12:
                warnings.warn("dx and dy focus differs")

            focus = (self.surfaces[-1].paraxial_center[2] +
                     0.5 * (dx + dy) * np.sign(self.surfaces[-1].input_axis[2]))
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

    def get_normal(self, pts: NDArray):
        """
        Get normal value at pts. Should be implemented so if pass a ray instead of a point, will
        take the (xo, yo, zo) part and ignore the rest

        :param pts:
        :return:
        """
        pass

    def get_intersect(self, rays: NDArray, material: Material):
        """
        Find intersection points between rays and Surface for rays propagating through a given Material,
        and return resulting rays at intersection

        :param rays:
        :param material:
        :return:
        """
        pass

    def propagate(self,
                  ray_array: NDArray,
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

    def get_seidel_third_order_fns(self):
        """

        :return:
        """
        raise NotImplementedError()

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
        # todo: helper function for calculating seidel aberrations (not finished)

        mat = self.get_ray_transfer_matrix(n1, n2)
        # idea: form full ABCD matrix as free prop in n2 * mat * free prop in n1, then set B = 0
        with np.errstate(divide="ignore"):
            if not np.isinf(s):
                sp_x = np.atleast_1d(-n2 * (-mat[0, 0] * s / n1 + mat[0, 1]) / np.array(-mat[1, 0] * s / n1 + mat[1, 1]))
                sp_y = np.atleast_1d(-n2 * (-mat[2, 2] * s / n1 + mat[2, 3]) / np.array(-mat[3, 2] * s / n1 + mat[3, 3]))
            else:
                sp_x = np.atleast_1d(-n2 * mat[0, 0] / mat[1, 0])
                sp_y = np.atleast_1d(-n2 * mat[2, 2] / mat[3, 2])

        return sp_x

    def is_pt_on_surface(self, pts: NDArray):
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
                  ray_array: NDArray,
                  material1: Material,
                  material2: Material) -> NDArray:
        """
        Given a set of rays, propagate them to the Surface of this object and compute their refraction
        using Snell's law. Return the update ray array with these two new ray positions

        :param ray_array: nsurfaces x nrays x 8
        :param material1: Material on first side of Surface
        :param material2: Material on second side of Surface
        """
        if ray_array.ndim == 1:
            ray_array = np.expand_dims(ray_array, axis=(0, 1))
        if ray_array.ndim == 2:
            ray_array = np.expand_dims(ray_array, axis=0)

        # get latest rays
        rays = ray_array[-1]
        rays_intersection = self.get_intersect(rays, material1)
        normals = self.get_normal(rays_intersection)

        # check ray was coming from the "front side"
        # i.e. the ray has to be coming from the correct 2*pi area of space
        ray_normals = rays[:, 3:6]
        cos_ray_input = np.sum(ray_normals * np.expand_dims(self.input_axis, axis=0), axis=1)
        with np.errstate(invalid="ignore"):
            not_incoming = cos_ray_input < 0
        rays_intersection[not_incoming] = np.nan

        # ######################
        # do refraction
        # ######################
        # rays_refracted = refract(rays_intersection, normals, material1, material2)
        ds = rays_intersection[:, 3:6]
        wls = np.expand_dims(rays_intersection[:, 7], axis=1)

        # basis for computation (normal, nb, nc)
        # nb orthogonal to normal and ray
        with np.errstate(invalid="ignore"):
            nb = np.cross(ds, normals)
            nb = nb / np.expand_dims(np.linalg.norm(nb, axis=1), axis=1)
            nb[np.isnan(nb)] = 0

            nc = np.cross(normals, nb)
            nc = nc / np.expand_dims(np.linalg.norm(nc, axis=1), axis=1)
            nc[np.isnan(nc)] = 0

            # snell's law
            # the tangential component (i.e. nc direction) of k*n*ds is preserved across the interface
            mag_nc = material1.n(wls) / material2.n(wls) * np.expand_dims(np.sum(nc * ds, axis=1), axis=1)
            sign_na = np.expand_dims(np.sign(np.sum(normals * ds, axis=1)), axis=1)
            # normalize outgoing ray direction. By construction nothing in nb direction
            ds_out = mag_nc * nc + sign_na * np.sqrt(1 - mag_nc ** 2) * normals

            rays_refracted = np.concatenate((rays_intersection[:, :3],
                                             ds_out,
                                             rays_intersection[:, 6:]), axis=1)
            rays_refracted[np.isnan(ds_out[:, 0]), :3] = np.nan

        # append refracted rays to full array
        ray_array = np.concatenate((ray_array,
                                    np.stack((rays_intersection,
                                              rays_refracted), axis=0)),
                                   axis=0)

        return ray_array


class ReflectingSurface(Surface):
    def propagate(self,
                  ray_array: NDArray,
                  material1: Material,
                  material2: Optional[Material] = None) -> NDArray:
        """
        Given a set of rays, propagate them to the Surface of this object and compute their reflection.
        Find new rays after reflecting off a Surface defined by a given normal using the law of reflection.
        Return the update ray array with these two new ray positions

        :param ray_array: nsurfaces x nrays x 8
        :param material1:
        :param material2:
        :return rays:
        """
        if ray_array.ndim == 1:
            ray_array = np.expand_dims(ray_array, axis=(0, 1))
        if ray_array.ndim == 2:
            ray_array = np.expand_dims(ray_array, axis=0)

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
        with np.errstate(invalid="ignore"):
            nb = np.cross(ds, normals)
            nb = nb / np.expand_dims(np.linalg.norm(nb, axis=1), axis=1)
            nb[np.isnan(nb)] = 0

            nc = np.cross(normals, nb)
            nc = nc / np.expand_dims(np.linalg.norm(nc, axis=1), axis=1)
            nc[np.isnan(nc)] = 0

        # law of reflection
        # the normal component (i.e. na direction) changes sign
        mag_na = -np.expand_dims(np.sum(normals * ds, axis=1), axis=1)
        mag_nc = np.expand_dims(np.sum(nc * ds, axis=1), axis=1)
        ds_out = mag_na * normals + mag_nc * nc

        rays_refracted = np.concatenate((rays_intersection[:, :3],
                                   ds_out,
                                   rays_intersection[:, 6:]),
                                  axis=1)
        rays_refracted[np.isnan(ds_out[:, 0]), :3] = np.nan

        # append these rays to full array
        ray_array = np.concatenate((ray_array,
                                    np.stack((rays_intersection,
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

    def get_normal(self, pts: NDArray):
        pts = np.atleast_2d(pts)
        normal = np.atleast_2d(self.normal)
        return np.tile(normal, (pts.shape[0], 1))

    def get_intersect(self, rays: NDArray, material: Material):
        rays_int, ts = propagate_ray2plane(rays,
                                           self.normal,
                                           self.center,
                                           material,
                                           exclude_backward_propagation=True)
        return rays_int

    def is_pt_on_surface(self, pts: NDArray):
        pts = np.atleast_2d(pts)
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
        xc, yc, zc = self.center
        nx, ny, nz = self.normal

        on_surface = np.abs((x - xc) * nx + (y - yc) * ny + (z - zc) * nz) < 1e-12

        return on_surface

    def get_ray_transfer_matrix(self, n1=None, n2=None):
        mat = np.array([[1, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 1]])
        return mat

    def get_seidel_third_order_fns(self, n1, n2, s, sp, t, tp, h, H):
        if s != 0:
            K = n1 * (-1 / s)  # abbe invariant for image/obj point
            c1 = (1 / (n2 * sp) - 1 / (n1 * s))
        else:  # in this case h = 0, and K only enters with a factor of h so doesn't matter...
            K = 0
            c1 = 0

        if t != 0:
            L = n1 * (-1 / t)  # abbe invariant for pupil/pupil image point
            c2 = (1 / (n2 * tp) - 1 / (n1 * t))
        else:  # in this case H = 0, and L only enters with a factor of H so doesn't matter...
            L = 0
            c2 = 0

        # these are B, F, C, D, E
        # i.e. spherical, coma, astigmatism and curvature of field, and distortion
        coeffs = np.array([0.5 * h ** 4 * K ** 2 * c1,
                           0.5 * H * h ** 3 * K * L * c1,
                           0.5 * H ** 2 * h ** 2 * L ** 2 * c1,
                           0.5 * H ** 2 * h ** 2 * (K * L * c1 - K * (K - L) * c2),
                           0.5 * H ** 3 * h * (L ** 2 * c1 - L * (K - L) * c2),
                           ])

        return coeffs

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

        # construct points on line using broadcasting
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

    def get_normal(self, pts: NDArray):
        pts = np.atleast_2d(pts)
        normal = np.atleast_2d(self.normal)
        return np.tile(normal, (pts.shape[0], 1))

    def get_intersect(self, rays: NDArray, material: Material):
        rays_int, ts = propagate_ray2plane(rays, self.normal, self.center, material)
        # exclude rays which will not intersect plane (but would have intersected in the past)
        rays_int[ts < 0] = np.nan

        return rays_int

    def is_pt_on_surface(self, pts: NDArray):
        pts = np.atleast_2d(pts)
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
        xc, yc, zc = self.center
        nx, ny, nz = self.normal

        on_surface = np.abs((x - xc) * nx + (y - yc) * ny + (z - zc) * nz) < 1e-12

        return on_surface

    def get_ray_transfer_matrix(self, n1: float, n2: float):
        mat = np.array([[1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, -1]])
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

    def get_normal(self, pts: NDArray):
        """
        Return the outward facing normal if self.aperture_radius > 0, otherwise the inward facing normal

        :param pts: each pt defined by row of matrix
        :return normals:
        """
        pts = np.atleast_2d(pts)[:, :3]
        normals = (pts - np.expand_dims(np.array(self.center), axis=0)) / self.radius
        return normals

    def get_intersect(self, rays: NDArray, material: Material) -> NDArray:
        rays = np.atleast_2d(rays)

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

        # we only want t > 0, since these are the forward points for the rays
        # and of the t > 0, we want the smallest t
        ts = np.stack((0.5 * (-B + np.sqrt(B**2 - 4 * A * C)),
                       0.5 * (-B - np.sqrt(B**2 - 4 * A * C))), axis=1)

        with np.errstate(invalid="ignore"):
            ts[ts < 0] = np.inf

        t_sol = np.min(ts, axis=1)
        t_sol[t_sol == np.inf] = np.nan

        pts = np.stack((xo, yo, zo), axis=1) + np.stack((dx, dy, dz), axis=1) * np.expand_dims(t_sol, axis=1)
        phase_shift = np.linalg.norm(pts - np.stack((xo, yo, zo), axis=1), axis=1) * 2 * np.pi / wls * material.n(wls)

        rays_int = np.concatenate((pts, np.stack((dx, dy, dz, phase_o + phase_shift, wls), axis=1)), axis=1)

        return rays_int

    def is_pt_on_surface(self, pts: NDArray) -> NDArray:
        """
        Check if point is on sphere surfaces

        :param pts:
        :return:
        """
        pts = np.atleast_2d(pts)
        diff = (pts[:, 0] - self.center[0]) ** 2 + (pts[:, 1] - self.center[1]) ** 2 + (pts[:, 2] - self.center[2]) ** 2
        on_surface = np.abs(diff - self.radius**2) < 1e-12
        return on_surface

    def get_seidel_third_order_fns(self, n1, n2, s, sp, t, tp, h, H):
        if s != 0:
            K = n1 * (1 / self.radius - 1 / s)  # abbe invariant for image/obj point
            c1 = (1 / (n2 * sp) - 1 / (n1 * s))
        else:  # in this case h = 0, and K only enters with a factor of h so doesn't matter...
            K = 0
            c1 = 0

        if t != 0:
            L = n1 * (1 / self.radius - 1 / t)  # abbe invariant for pupil/pupil image point
            c2 = (1 / (n2 * tp) - 1 / (n1 * t))
        else:  # in this case H = 0, and L only enters with a factor of H so doesn't matter...
            L = 0
            c2 = 0

        # B, F, C, D, E
        # i.e. spherical, coma, astigmatism and curvature of field, and distortion
        coeffs = np.array([0.5 * h**4 * K**2 * c1,
                           0.5 * H * h**3 * K * L * c1,
                           0.5 * H**2 * h**2 * L**2 * c1,
                           0.5 * H**2 * h**2 * (K * L * c1 - K * (K - L) * c2),
                           0.5 * H**3 * h * (L**2 * c1 - L * (K - L) * c2),
                           ])

        return coeffs

    def get_ray_transfer_matrix(self, n1: float, n2: float) -> NDArray:
        # test if we are going from "inside the sphere" to "outside the sphere" i.e. ray is striking the concave side
        # or the other way
        pc_to_c = self.center - self.paraxial_center
        sgn = np.sign(np.dot(pc_to_c, self.input_axis))

        with np.errstate(divide="ignore"):
            f = sgn * np.abs(self.radius) / np.array(n2 - n1)

        mat = np.array([[1,    0, 0,    0, 0],
                        [-1/f, 1, 0,    0, 0],
                        [0,    0, 1,    0, 0],
                        [0,    0, -1/f, 1, 0],
                        [0,    0, 0,    0, 1]])
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

    def get_normal(self):
        pass

    def get_intersect(self, rays: NDArray, material: Material) -> NDArray:
        rays_int, ts = propagate_ray2plane(rays, self.normal, self.center, material)
        with np.errstate(invalid="ignore"):
            rays_int[ts < 0] = np.nan
        return rays_int

    def is_pt_on_surface(self, pts: NDArray) -> NDArray:
        pts = np.atleast_2d(pts)
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
        xc, yc, zc = self.center
        nx, ny, nz = self.normal

        on_surface = np.abs((x - xc) * nx + (y - yc) * ny + (z - zc) * nz) < 1e-12

        return on_surface

    def propagate(self,
                  rays: NDArray,
                  material1: Material,
                  material2: Material) -> NDArray:
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

        if rays.ndim == 1:
            rays = np.expand_dims(rays, axis=(0, 1))
        elif rays.ndim == 2:
            rays = np.expand_dims(rays, axis=0)

        wls = rays[-1, :, 7]

        # #####################################
        # get the three surfaces we will need in our calculation: front focal plane (i.e. before the lens),
        # lens Surface, back focal plane (i.e. after the lens)
        # #####################################
        front_focal_pts = (np.expand_dims(self.center, axis=0) -
                           np.expand_dims(self.normal, axis=0) * self.focal_len *
                           np.expand_dims(material1.n(wls), axis=1))
        back_focal_pts = (np.expand_dims(self.center, axis=0) +
                          np.expand_dims(self.normal, axis=0) * self.focal_len *
                          np.expand_dims(material2.n(wls), axis=1))

        # #####################################
        # find position rays intersect the object plane (front focal plane)
        # if rays are already in front of the object plane, propagate them backwards to reach it
        # #####################################
        rays_ffp, _ = propagate_ray2plane(rays[-1],
                                          self.normal,
                                          front_focal_pts,
                                          material1,
                                          exclude_backward_propagation=False)

        # #####################################
        # compute geometric data (height and angle) of rays in ffp relative to the lens axis
        # #####################################

        # get unit vectors pointing along input rays
        s1 = rays_ffp[:, 3:6]
        # n0 are the unit vectors pointing along rays after projecting out the portion along the lens normal
        ray_normal_dot = np.sum(s1 * np.expand_dims(self.normal, axis=0), axis=1)
        s1_perp_uvec = s1 - np.expand_dims(ray_normal_dot, axis=1) * np.expand_dims(self.normal, axis=0)

        # only normalize non-zero rays
        with np.errstate(invalid="ignore"):
            # to_normalize = np.sum(np.abs(n0), axis=1) > 1e-13
            # n0[to_normalize] = n0[to_normalize] / np.expand_dims(np.linalg.norm(n0[to_normalize], axis=1), axis=1)
            s1_perp_norm = np.linalg.norm(s1_perp_uvec, axis=1)
            to_normalize = s1_perp_norm > 1e-12
            s1_perp_uvec[to_normalize] = s1_perp_uvec[to_normalize] / np.expand_dims(s1_perp_norm[to_normalize], axis=1)

        # compute the vectorial "height" of rays above optical axis
        # r1_vec is the vector in the FFP pointing to the position of the ray
        # i.e. the ray position after projecting out the optical axis direction
        r1_vec = rays_ffp[:, 0:3] - front_focal_pts
        r1_norm = np.linalg.norm(r1_vec, axis=1)

        # get unit vector
        to_normalize = r1_norm != 0
        r1_uvec = np.array(r1_vec, copy=True)
        r1_uvec[to_normalize] = r1_uvec[to_normalize] / np.expand_dims(np.linalg.norm(r1_uvec[to_normalize],
                                                                                      axis=1),
                                                                       axis=1)

        # sine of angle between incoming ray and the optical axis
        sin_t1 = np.sum(s1_perp_uvec * s1, axis=1)

        # #####################################
        # construct rays as they appear in the BFP (i.e. after the lens)
        # #####################################
        rays_bfp = np.zeros(rays_ffp.shape)

        # keep same wavelengths
        rays_bfp[:, 7] = wls

        # compute ray positions in BFP. These depend only on the input direction and are found by
        # vectorial position = h * n0 where h = n*fl*sin(theta_1)
        h2 = np.expand_dims(material1.n(wls), axis=1) * self.focal_len * np.expand_dims(sin_t1, axis=1) * s1_perp_uvec
        rays_bfp[:, :3] = h2 + np.expand_dims(back_focal_pts, axis=0)

        # output angles
        # get unit vector for input positions, r0
        with np.errstate(invalid="ignore"):
            sin_t2 = -r1_norm / self.focal_len / material2.n(wls)
            cos_t2 = np.sqrt(1 - sin_t2**2)
            rays_bfp[:, 3:6] = np.expand_dims(sin_t2, axis=1) * r1_uvec + \
                               np.expand_dims(cos_t2, axis=1) * np.expand_dims(self.normal, axis=0)

        # #####################################
        # exclude points which are outside the NA of the lens
        # #####################################
        with np.errstate(invalid="ignore"):
            input_angle_too_steep = np.abs(sin_t1) > np.sin(self.alpha)
            output_angle_too_steep = np.abs(sin_t2) > np.sin(self.alpha)
            rays_bfp[np.logical_or(input_angle_too_steep, output_angle_too_steep)] = np.nan

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
        plane_wave_phase = np.sum(r1_vec * s1, axis=1)

        rays_bfp[:, 6] = rays_ffp[:, 6] - \
                         2 * np.pi / wls * material1.n(wls) * plane_wave_phase + \
                         2 * np.pi / wls * (material1.n(wls)**2 * self.focal_len + material2.n(wls)**2 * self.focal_len)

        # #####################################
        # propagate rays from bfp backwards to lens position
        # note: from this direction don't care about aperture
        # #####################################
        rays_after_lens, _ = propagate_ray2plane(rays_bfp,
                                                 self.normal,
                                                 self.center,
                                                 material2,
                                                 exclude_backward_propagation=False)

        # also need position that the rays would normally intersect lens position
        rays_before_lens, _ = propagate_ray2plane(rays[-1],
                                                  self.normal,
                                                  self.center,
                                                  material1)

        # #####################################
        # output ray array is all the rays that were passed in and two new Surface
        # (1) right before the lens and (2) right after
        # #####################################
        rays_out = np.concatenate((rays, np.stack((rays_before_lens, rays_after_lens), axis=0)), axis=0)

        return rays_out

    def get_ray_transfer_matrix(self, n1: float, n2: float) -> NDArray:
        mat = np.array([[1,                 0, 0,                 0, 0],
                        [-1/self.focal_len, 1, 0,                 0, 0],
                        [0,                 0, 1,                 0, 0],
                        [0,                 0, -1/self.focal_len, 1, 0],
                        [0,                 0, 0,                 0, 1]])
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
