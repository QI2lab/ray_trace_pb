"""
Ray trace through spherical particle

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
ensuring the coordinate system is right-handed
"""
import copy
import numpy as np
import matplotlib.pyplot as plt
import warnings

# analyze optical systems
def ray_trace_system(rays, surfaces, materials):
    """
    race rays through a system of diffractive optical elements

    @param rays: N x 8 array
    @param list surfaces: list of surfaces
    @param list[float] materials: indices of refraction between surfaces. If there are Ns surfaces,
     this should have Ns + 1 elements. The first element is the index of refraction before the first
     surface of the system, and the last element is the index of refraction after the last surface of the system.
    @return rays_out: M x N x 8 array, where we represent the complete raytracing by M positions of each ray
    """
    # raise DeprecationWarning("use System() instead")

    if len(materials) != len(surfaces) + 1:
        raise ValueError("length of materials should be len(surfaces) + 1")

    rays = np.array(rays)

    for ii in range(len(surfaces)):
        rays = surfaces[ii].propagate(rays, materials[ii], materials[ii + 1])

    return rays


def compute_paraxial_matrix(surfaces, materials, wavelength, initial_distance=0, final_distance=0):
    """
    # todo: instead of indices of refraction use a material ... and derive index of refraction from material + wavelength
    Generate the ray transfer (ABCD) matrix for an optical system

    Assume that the optical system starts at the provided initial distance before the first surface

    @param initial_distance: distance from the ray launching surface to the first surface in the optical system
    @param surfaces: list of surfaces in the optical system
    @param indices_of_refraction: indices of refraction between surfaces
    @return abcd_mat:
    """
    mats = [] # helpful for debugging

    indices_of_refraction = [m.n(wavelength) for m in materials]

    # propagate initial distance
    ray_xfer_mat = get_free_space_abcd(initial_distance, n=indices_of_refraction[0])
    mats.append(ray_xfer_mat)

    for ii in range(len(surfaces)):
        # apply ray transfer matrix for current surface
        abcd_temp = surfaces[ii].get_ray_transfer_matrix(indices_of_refraction[ii], indices_of_refraction[ii + 1])
        ray_xfer_mat = abcd_temp.dot(ray_xfer_mat)
        mats.append(abcd_temp)

        # apply ray transfer matrix propagating between surfaces
        if ii < (len(surfaces) - 1):
            dist = np.linalg.norm(surfaces[ii].paraxial_center - surfaces[ii + 1].paraxial_center)
            abcd_prop = get_free_space_abcd(dist, n=indices_of_refraction[ii+1])
            ray_xfer_mat = abcd_prop.dot(ray_xfer_mat)
            mats.append(abcd_prop)

    # propagate final distance
    ray_xfer_temp = get_free_space_abcd(final_distance, n=indices_of_refraction[-1])
    ray_xfer_mat = ray_xfer_temp.dot(ray_xfer_mat)
    mats.append(ray_xfer_temp)

    return ray_xfer_mat


def get_reversed_system(surfaces, materials):
    """
    Create a new optical system where the order of the surfaces is reversed.

    :param list surfaces:
    :param list n_rev:
    :return surfaces_rev, n_rev:
    """
    # raise DeprecationWarning("use System() instead")
    surfaces_rev = [copy.deepcopy(surfaces[-ii]) for ii in range(1, len(surfaces) + 1)]

    for ii in range(len(surfaces)):
        surfaces_rev[ii].input_axis *= -1
        surfaces_rev[ii].output_axis *= -1

    n_rev = [materials[-ii] for ii in range(1, len(materials) + 1)]
    return surfaces_rev, n_rev


def compute_third_order_seidel(surfaces, materials, wavelength):
    # first check where the aperture stop is
    is_ap_stop = [s.is_aperture_stop for s in surfaces]

    if not np.any(is_ap_stop):
        raise ValueError("none of the surfaces were labelled as the aperture stop")
    if np.sum(is_ap_stop) > 1:
        raise ValueError("more than one of the surfaces was labelled as the aperture stop")

    istop = int(np.where(is_ap_stop)[0])

    # find entrance pupil
    surfaces_before_stop = surfaces[:istop + 1]
    materials_before_stop = materials[:istop + 2]

    surfaces_before_stop, materials_before_stop = get_reversed_system(surfaces_before_stop, materials_before_stop)

    # to avoid refracting through the stop again, modify the index of refraction
    materials_before_stop[0] = materials_before_stop[1]

    s_temp, _ = auto_focus(surfaces_before_stop, materials_before_stop, wavelength, mode="paraxial-focused")
    # correct entrance pupil directions
    entrance_pupil, _ = get_reversed_system([s_temp[-1]], [1, 1])
    entrance_pupil = entrance_pupil[0]

    # find exit pupil
    # surfaces_after_stop = surfaces[istop:]
    # ns_after_stop = materials[istop - 1:]
    #
    # s_temp, _ = auto_focus(surfaces_after_stop, ns_after_stop, mode="paraxial")
    # exit_pupil = s_temp[-1]

    # needed quantities
    nsurfaces = len(surfaces)
    aberrations_3rd = np.zeros((nsurfaces, 5))
    s = np.zeros(nsurfaces) # object points per surface
    sp = np.zeros(nsurfaces) # image points per surface
    h = np.zeros(nsurfaces) # h1 = s1 / (t1 - s1) where t = distance from entrance pupil to surface vertex
    H = np.zeros(nsurfaces) # H1 = t1 / no
    t = np.zeros(nsurfaces)
    tp = np.zeros(nsurfaces)
    d = np.zeros(nsurfaces - 1)

    # initialize values for first surface
    s[0] = surfaces[0].paraxial_center[2]
    sp[0] = surfaces[0].solve_img_eqn(s[0], materials[0], materials[1])
    t[0] = entrance_pupil.paraxial_center[2] - surfaces[0].paraxial_center[2]
    tp[0] = surfaces[0].solve_img_eqn(t[0], materials[0], materials[1])
    h[0] = s[0] / (t[0] - s[0])
    H[0] = t[0] / materials[0]

    # compute subsequent values of image positions and pupil positions
    for ii in range(1, len(h)):
        # d[ii] is the distance between ii and ii + 1 surfaces
        d[ii - 1] = surfaces[ii].paraxial_center[2] - surfaces[ii - 1].paraxial_center[2]

        # new object is previous image relative to new surfaces
        s[ii] = sp[ii - 1] - d[ii - 1]
        sp[ii] = surfaces[ii].solve_img_eqn(s[ii], materials[ii], materials[ii + 1])
        # new pupil is previous image relative to new surfaces
        t[ii] = tp[ii - 1] - d[ii - 1]
        tp[ii] = surfaces[ii].solve_img_eqn(t[ii], materials[ii], materials[ii + 1])

        # see Born and Wolf chapter 5.5 eq's (9) and (16)
        # h[ii] = h[ii - 1] * s[ii] / sp[ii - 1] # recursion has problem if one is zero...
        h[ii] = s[ii] / (t[ii] - s[ii])
        #H[ii] = H[ii - 1] * t[ii] / tp[ii - 1]
        H[ii] = t[ii] / materials[ii]

    # solve for aberrations
    for ii in range(nsurfaces):
        aberrations_3rd[ii] = surfaces[ii].get_seidel_third_order_fns(
                                            materials[ii], materials[ii + 1],
                                            s[ii], sp[ii], t[ii], tp[ii], h[ii], H[ii])

    return aberrations_3rd


# helper methods for finding focus, cardinal points, etc.
def auto_focus(surfaces, materials, wavelength, mode="ray-fan"):
    """
    Perform auto-focus operation. This function can handle rays which are initially collimated or initially diverging

    # todo: handle case where optical system extends past focus

    :param list surfaces: list of surfaces, which should start with the surfaces the rays are incident from
    :param materials: list of indices of refraction betweeen surfaces
    :param wavelength:
    :param mode: "ray-fan", "collimated", "paraxial-focused", or "paraxial-collimated"
    :return updated_surfaces, updated_n:
    """
    if mode == "ray-fan":
        # todo: maybe take sequence of rays with smaller and smaller angles...
        rays_focus = get_ray_fan([0, 0, 0], 1e-9, 3, wavelength)
        rays_focus = ray_trace_system(rays_focus, surfaces, materials)
        focus = intersect_rays(rays_focus[-1, 1], rays_focus[-1, 2])[0, 2]
    elif mode == "collimated":
        rays_focus = get_collimated_rays([0, 0, 0], 1e-9, 3, wavelength)
        rays_focus = ray_trace_system(rays_focus, surfaces, materials)
        focus = intersect_rays(rays_focus[-1, 1], rays_focus[-1, 2])[0, 2]

    elif mode == "paraxial-focused":
        ns = [m.n(wavelength) for m in materials]
        abcd = compute_paraxial_matrix(surfaces, ns, initial_distance=0, final_distance=0)
        # determine what free space propagation matrix we need such that initial ray (0, n*theta) -> (0, n'*theta')
        dx = -abcd[0, 1] / abcd[1, 1] * ns[-1]
        dy = -abcd[2, 3] / abcd[3, 3] * ns[-1]

        if np.abs(dx - dy) >= 1e-12:
            warnings.warn("dx and dy focus differs")

        focus = surfaces[-1].paraxial_center[2] + 0.5 * (dx + dy) * np.sign(surfaces[-1].input_axis[2])
    elif mode == "paraxial-collimated":
        ns = [m.n(wavelength) for m in materials]
        abcd = compute_paraxial_matrix(surfaces, ns, initial_distance=0, final_distance=0)
        # determine what free space propagation matrix we need such that initial ray (h, n*theta) -> (0, n'*theta')
        dx = -abcd[0, 0] / abcd[1, 0] * ns[-1]
        dy = -abcd[2, 2] / abcd[3, 2] * ns[-1]

        if np.abs(dx - dy) >= 1e-12:
            warnings.warn("dx and dy focus differs")

        focus = surfaces[-1].paraxial_center[2] + 0.5 * (dx + dy) * np.sign(surfaces[-1].input_axis[2])
    else:
        raise ValueError(f"mode must be 'ray-fan', or 'collimated' 'paraxial-focused', or paraxial-collimated' but was '{mode:s}'")

    updated_surfaces = surfaces + [flat_surface([0, 0, focus], surfaces[-1].input_axis, surfaces[-1].aperture_rad)]
    updated_n = materials + [materials[-1]]

    return updated_surfaces, updated_n


def find_paraxial_focus(abcd_mat, n=1):
    """
    Find paraxial focus of a paraxial optical system from its ray-tarnsfer matrix

    @param abcd_mat:
    @param n: index of refraction of final medium
    @return dx, abcd_mat_x, efl_x, dy, abcd_mat_y, efl_y:
    """

    # if I left multiply my ray transfer matrix by free space matrix, then combined matrix has lens/focal form
    # for certain distance of propagation dx. Find this by setting A + d/n * C = 0
    dx = -abcd_mat[0, 0] / abcd_mat[1, 0] * n
    dy = -abcd_mat[2, 2] / abcd_mat[3, 2] * n

    abcd_mat_x = get_free_space_abcd(dx, n).dot(abcd_mat)
    abcd_mat_y = get_free_space_abcd(dy, n).dot(abcd_mat)

    # can also find the principal plane with the following construction
    # take income ray at (h1, n1*theta1 = 0). Consider the ray-transfer matrix which combines the optic and the
    # distance travelled dx to the focus
    # then extend the corresponding ray at (h2=0, n2*theta2) backwards until it reaches height h
    # can check this happens at position P2 = f2 + h1/theta2 (where theta2<0 here)
    # by construction the ray-transfer matrix above has A=0, but in any case we have the relationship C*h1 = n2*theta2
    # or P2 = f2 + n2/C -> EFL2 = f2 - P2 = -n2/C

    # todo: think it is wrong to call this the EFL
    # EFL = 1 / C, and this is not affect by right or left-multiplying the ray-transfer matrix by free space propagation
    efl_x = -n / abcd_mat_x[1, 0]
    efl_y = -n / abcd_mat_y[3, 2]

    return dx, abcd_mat_x, efl_x, dy, abcd_mat_y, efl_y


def find_cardinal_points(surfaces, materials, wavelength):
    """
    Compute cardinal points from ray tracing very small angles

    # todo: this fails if BFP is within optical system
    # todo: probably better to write with ABCD matrices...

    :param list surfaces: list of surfaces
    :param list materials: list of indices of refraction between surfaces
    :param float wavelength: wavelength
    :return f1, f2, pp1, pp2, efl1, efl2:
    """

    # find focal point to right of lens
    rays = np.array([[0, 0, 0, 0, 0, 1, 0, wavelength],
                     [1e-9, 0, 0, 0, 0, 1, 0, wavelength]])
    rays = ray_trace_system(rays, surfaces, materials)
    f2 = intersect_rays(rays[-1, 0], rays[-1, 1])

    # find focal point to left of lens
    surfaces_rev, n_rev = get_reversed_system(surfaces, materials)
    # todo: need to ensure rays past last surface...
    rays = np.array([[0, 0, 1e4, 0, 0, -1, 0, wavelength],
                     [1e-9, 0, 1e4, 0, 0, -1, 0, wavelength]])
    rays = ray_trace_system(rays, surfaces_rev, n_rev)
    f1 = intersect_rays(rays[-1, 0], rays[-1, 1])

    # propagate rays from front focal point to collimated to find first principal plane
    rays_fwd = get_ray_fan(f1, 1e-9, 3, wavelength)
    rays_fwd = ray_trace_system(rays_fwd, surfaces, materials)
    pt1 = intersect_rays(rays_fwd[0, 2], rays_fwd[-1, 2])
    pp1 = pt1[0, 2]

    # effective focal length
    efl1 = pp1 - f1[0, 2]

    # propagate rays backward from the back focal point to collimated to find the second principal plane
    surfaces_rev, n_rev = get_reversed_system(surfaces, materials)

    rays_back = get_ray_fan(f2, 1e-9, 3, wavelength, center_ray=(0, 0, -1))
    rays_back = ray_trace_system(rays_back, surfaces_rev, n_rev)
    pt2 = intersect_rays(rays_back[0, 2], rays_back[-1, 2])
    pp2 = pt2[0, 2]

    efl2 = f2[0, 2] - pp2

    return f1, f2, pp1, pp2, efl1, efl2


def find_paraxial_collimated_distance(mat1, mat2, n):
    """
    Given two sets of surfaces (e.g. two lenses) determine the distance which should be inserted between them
    to give a system which converts collimated rays to collimated rays

    @param mat1:
    @param mat2:
    @param n: index of refraction of intervening medium
    @return distance:
    """

    dx = -(mat1[0, 0] / mat1[1, 0] + mat2[1, 1] / mat2[1, 0]) * n
    dy = -(mat1[2, 2] / mat1[3, 2] + mat2[3, 3] / mat2[3, 2]) * n

    return dx, dy


# propagation and refraction
def get_free_space_abcd(d, n=1):
    """
    Compute the ray-transfer (ABCD) matrix for free space beam propagation
    @param d: distance beam propagated
    @param n: index of refraction
    @return mat:
    """
    mat = np.array([[1, d/n, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, d/n, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1]])
    return mat


def refract(rays, normals, material1, material2):
    """
    Refracts rays at surface with given normal by applying Snell's law

    :param rays: N x 8 array
    :param normals: N x 3 array
    :param n1: index of refraction on the side of the interface the rays are travelling from
    :param n2: index of refraction on the other side of the interface
    :param rays_out: N x 8 array
    """
    normals = np.atleast_2d(normals)
    rays = np.atleast_2d(rays)

    ds = rays[:, 3:6]
    wls = np.expand_dims(rays[:, 7], axis=1)

    # basis for computation (na, nb, nc)
    # na is normal direction, nb orthogonal to normal and ray
    na = normals

    with np.errstate(invalid="ignore"):
        nb = np.cross(ds, normals)
        nb = nb / np.expand_dims(np.linalg.norm(nb, axis=1), axis=1)
        nb[np.isnan(nb)] = 0

        nc = np.cross(na, nb)
        nc = nc / np.expand_dims(np.linalg.norm(nc, axis=1), axis=1)
        nc[np.isnan(nc)] = 0

        # snell's law
        # the tangential component (i.e. nc direction) of k*n*ds is preserved across the interface
        mag_nc = material1.n(wls) / material2.n(wls) * np.expand_dims(np.sum(nc * ds, axis=1), axis=1)
        sign_na = np.expand_dims(np.sign(np.sum(na * ds, axis=1)), axis=1)
        # normalize outgoing ray direction. By construction nothing in nb direction
        ds_out = mag_nc * nc + sign_na * np.sqrt(1 - mag_nc**2) * na

        rays_out = np.concatenate((rays[:, :3], ds_out, rays[:, 6:]), axis=1)
        rays_out[np.isnan(ds_out[:, 0]), :3] = np.nan

    return rays_out


def reflect(rays, normals):
    """
    Find new rays after reflecting off a surface defined by a given normal using the law of reflection

    @param rays: nrays x 8 array
    @param normals: array of size 3, in which case this normal is applied to all rays, or size nrays x 3, in
    which case each normal is applied to the corresponding ray
    @return rays_out: nrays x 8 array
    """
    normals = np.atleast_2d(normals)
    rays = np.atleast_2d(rays)

    ds = rays[:, 3:6]

    # basis for computation (na, nb, nc)
    # na is normal direction, nb orthogonal to normal and ray
    na = normals

    with np.errstate(invalid="ignore"):
        nb = np.cross(ds, normals)
        nb = nb / np.expand_dims(np.linalg.norm(nb, axis=1), axis=1)
        nb[np.isnan(nb)] = 0

        nc = np.cross(na, nb)
        nc = nc / np.expand_dims(np.linalg.norm(nc, axis=1), axis=1)
        nc[np.isnan(nc)] = 0

    # law of reflection
    # the normal component (i.e. na direction) changes sign
    mag_na = -np.expand_dims(np.sum(na * ds, axis=1), axis=1)
    mag_nc = np.expand_dims(np.sum(nc * ds, axis=1), axis=1)
    ds_out = mag_na * na + mag_nc * nc

    rays_out = np.concatenate((rays[:, :3], ds_out, rays[:, 6:]), axis=1)
    rays_out[np.isnan(ds_out[:, 0]), :3] = np.nan

    return rays_out


# tools for creating ray fans, manipulating rays, etc.
def get_ray_fan(pt, theta_max, n_thetas, wavelengths, nphis=1, center_ray=(0, 0, 1)):
    """
    Get fan of rays emanating from pt

    :param pt: [cx, cy, cz]
    :param float theta_max: maximum angle in radians
    :param int n_thetas: number of rays at different angles on axis
    :param float wavelength:
    :param int nphis: number of points of rotation about the optical axis. If nphis = 1, all rays will be in the plane
    :param center_ray:
    """

    # consider the central ray in direction no. Construct a n orthonormal basis from enx = y x no, eny = no x enx
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


def get_collimated_rays(pt, displacement_max, n_disps, wavelengths,
                        nphis=1, phi_start=0., normal=(0, 0, 1)):
    """
    Get a fan of collimated arrays along a certain direction. The rays will be generated in a plane
    with normal along this direction, which will generally not be perpendicular to the optical axis.

    Note that this approach avoids the need to know what the index of refraction of the medium is

    @param pt: point in the origin plane
    @param displacement_max: maximum radial displacement
    @param n_disps: number of displacements
    @param wavelengths: either floating point or an array the same size as n_disps * nphis
    @param nphis: number of rays in azimuthal direction
    @param phi_start: angle about normal to start at
    @param normal: normal of plane
    @return rays:
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


def intersect_rays(ray1, ray2):
    """
    Find intersection point between two rays, assuming free space propagation

    if either s or t is negative then these rays previously intersected

    @param ray1:
    @param ray2:
    @return intersection_pt:
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
    t = np.zeros(len(ray1))* np.nan
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


def propagate_ray2plane(rays, normal, center, material, exclude_backward_propagation=False):
    """
    Find intersection between rays and a plane. Plane is defined by a normal vector and a point on the
    plane

    :param rays: N x 8 array
    :param normal: normal of the plane. Should be broadcastable to the shape N x 3
    :param center: point on the plane. Should be broadcastable to the shape N x 3
    :param material: material through which rays are propagating
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


def ray_angle_about_axis(rays, reference_axis):
    """
    Given a set of rays, compute their angles relative to a given axis, and compute the orthogonal direction to the
    axis which the ray travels in
    """
    rays = np.atleast_2d(rays)

    cosines = np.sum(rays[:, 3:6] * np.expand_dims(reference_axis, axis=0), axis=1)
    angles = np.arccos(cosines)
    na = rays[:, 3:6] - np.expand_dims(cosines, axis=1) * np.expand_dims(reference_axis, axis=0)
    na = na / np.expand_dims(np.linalg.norm(na, axis=1), axis=1)

    return angles, na


def dist_pt2plane(pts, normal, center):
    """
    Calculate minimum distance between points and a plane defined by normal and center
    """
    pts = np.atleast_2d(pts)
    npts = pts.shape[0]

    rays = np.concatenate((pts, np.tile(normal, (npts, 1)), np.zeros((npts, 2))), axis=1)
    rays_int, _ = propagate_ray2plane(rays, normal, center, vacuum())

    dists = np.linalg.norm(rays_int[:, :3] - pts, axis=1)
    nearest_pts = rays_int[:, :3]

    return dists, nearest_pts


# display
def plot_rays(ray_array, surfaces: list = None, phi: float = 0, colors: list = None, label=None, ax=None, **kwargs):
    """
    Plot rays and optical surfaces

    @param ray_array: nsurfaces X nrays x 8
    @param surfaces: list of surfaces
    @param phi: angle describing the azimuthal plane to plot. phi = 0 gives the meridional/tangential plane while
    phi = pi/2 gives the sagittal plane. # todo: not implemented for drawing the surface projections
    @param colors: list of colors to plot rays
    @param ax: axis to plot results on. If None, a new figure will be generated
    @param kwargs: passed through to figure, if it does not already exist
    @return fig_handle, axis:
    """
    raise DeprecationWarning("Use System() instead")

    if ax is None:
        figh = plt.figure(**kwargs)
        ax = plt.subplot(1, 1, 1)
    else:
        figh = ax.get_figure()


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

    ax.set_xlabel("z-position")
    ax.set_ylabel("height")

    if surfaces is not None:
        for s in surfaces:
            s.draw(ax)

    return figh, ax


def plot_spot_diagram(rays, **kwargs):
    """
    Todo: think spot diagram should have points equally spaced in the aperture
    """
    figh = plt.figure(**kwargs)
    figh.suptitle("Spot diagram")
    ax = figh.add_subplot(1, 1, 1)

    ax.plot(rays[:, 1], rays[:, 0], "b.")
    ax.set_xlabel("y-position")
    ax.set_ylabel("x-position")

    ax.axis("equal")

    return figh, ax

# ################################################
# collections of optical elements
# ################################################
class system:
    """
    Collection of surfaces
    """

    def __init__(self, surfaces, materials):
        """

        @param surfaces: length n
        @param ns: length n-1
        """
        if len(materials) != (len(surfaces) - 1):
            raise ValueError()

        self.surfaces = surfaces
        self.materials = materials

    def reverse(self):
        """
        flip direction of the optic we are considering (so typically rays now enter from the right)
        @return:
        """
        surfaces_rev = [copy.deepcopy(self.surfaces[-ii]) for ii in range(1, len(self.surfaces) + 1)]

        for ii in range(len(self.surfaces)):
            surfaces_rev[ii].input_axis *= -1
            surfaces_rev[ii].output_axis *= -1

        materials_rev = [self.materials[-ii] for ii in range(1, len(self.materials) + 1)]

        return system(surfaces_rev, materials_rev)

    def concatenate(self, other, material, distance=0, axis=(0, 0, 1)):
        """
        add another optic after this one
        @param other:
        @param material:
        @param distance:
        @param axis:
        @return:
        """
        # specify distance between surfaces as distances between the paraxial foci

        new_surfaces = [copy.deepcopy(s) for s in other.surfaces]
        for ii, s in enumerate(new_surfaces):
            # C_i(new) = C_{i-1}(new) + [C_i(old) - C_{i-1}(old)]
            if ii == 0:
                shift = self.surfaces[-1].paraxial_center + distance * np.array(axis) - s.paraxial_center
            else:
                shift = new_surfaces[ii - 1].paraxial_center - other.surfaces[ii - 1].paraxial_center

            s.center += shift
            s.paraxial_center += shift

        s = self.surfaces + new_surfaces
        materials = self.materials + [material] + other.materials
        return system(s, materials)

    def ray_trace(self, rays, input_medium, output_medium):
        """
        ray trace through optical system
        @param rays:
        @param input_medium:
        @param output_medium:
        @return:
        """
        materials = [input_medium] + self.materials + [output_medium]

        if len(materials) != len(self.surfaces) + 1:
            raise ValueError("length of materials should be len(surfaces) + 1")

        rays = np.array(rays)

        for ii in range(len(self.surfaces)):
            rays = self.surfaces[ii].propagate(rays, materials[ii], materials[ii + 1])

        return rays

    def compute_paraxial_matrix(self, wavelength, initial_material, final_material):
        """
        # todo: instead of indices of refraction use a material ... and derive index of refraction from material + wavelength
        Generate the ray transfer (ABCD) matrix for an optical system

        Assume that the optical system starts at the provided initial distance before the first surface

        @param initial_distance: distance from the ray launching surface to the first surface in the optical system
        @param surfaces: list of surfaces in the optical system
        @param indices_of_refraction: indices of refraction between surfaces
        @return abcd_mat:
        """
        mats = []  # helpful for debugging

        surfaces = self.surfaces
        materials = [initial_material] + self.materials + [final_material]

        indices_of_refraction = [m.n(wavelength) for m in materials]

        # propagate initial distance
        ray_xfer_mat = get_free_space_abcd(0, n=indices_of_refraction[0])
        mats.append(ray_xfer_mat)

        for ii in range(len(surfaces)):
            # apply ray transfer matrix for current surface
            abcd_temp = surfaces[ii].get_ray_transfer_matrix(indices_of_refraction[ii], indices_of_refraction[ii + 1])
            ray_xfer_mat = abcd_temp.dot(ray_xfer_mat)
            mats.append(abcd_temp)

            # apply ray transfer matrix propagating between surfaces
            if ii < (len(surfaces) - 1):
                dist = np.linalg.norm(surfaces[ii].paraxial_center - surfaces[ii + 1].paraxial_center)
                abcd_prop = get_free_space_abcd(dist, n=indices_of_refraction[ii + 1])
                ray_xfer_mat = abcd_prop.dot(ray_xfer_mat)
                mats.append(abcd_prop)

        return ray_xfer_mat
    def get_cardinal_points(self, wavelength, initial_material, final_material):
        """

        @param wavelength:
        @param initial_material:
        @param final_material:
        @return fp1, fp2, pp1, pp2, efl1, efl2:
        """
        # todo: add nodal planes

        # find focal point to the right of the lens
        ray_xfer = self.compute_paraxial_matrix(wavelength, initial_material, final_material)
        d2x, _, efl2_x, d2y, _, efl2_y = find_paraxial_focus(ray_xfer, final_material.n(wavelength))
        d2 = 0.5 * (d2x + d2y)
        efl2 = 0.5 * (efl2_x + efl2_y)

        fp2 = self.surfaces[-1].paraxial_center + d2 * self.surfaces[-1].output_axis

        # find principal plane to the right
        pp2 = fp2 - efl2 * self.surfaces[-1].output_axis

        # find focal point to the left of the lens
        ray_xfer_inv = self.reverse().compute_paraxial_matrix(wavelength, final_material, initial_material)
        d1x, _, efl1_x, d1y, _, efl1_y = find_paraxial_focus(ray_xfer_inv, initial_material.n(wavelength))
        d1 = 0.5 * (d1x + d1y)
        efl1 = 0.5 * (efl1_x + efl1_y)
        fp1 = self.surfaces[0].paraxial_center - d1 * self.surfaces[0].input_axis

        # find principal plane to the left of the lens
        pp1 = fp1 + efl1 * self.surfaces[0].input_axis

        return fp1, fp2, pp1, pp2, efl1, efl2

    def plot(self, ray_array=None, phi: float = 0, colors: list = None, label = None, ax = None, ** kwargs):
        """
        Plot rays and optical surfaces

        @param ray_array: nsurfaces X nrays x 8
        @param surfaces: list of surfaces
        @param phi: angle describing the azimuthal plane to plot. phi = 0 gives the meridional/tangential plane while
        phi = pi/2 gives the sagittal plane. # todo: not implemented for drawing the surface projections
        @param colors: list of colors to plot rays
        @param ax: axis to plot results on. If None, a new figure will be generated
        @param kwargs: passed through to figure, if it does not already exist
        @return fig_handle, axis:
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

            ax.set_xlabel("z-position")
            ax.set_ylabel("height")

        # plot surfaces
        if self.surfaces is not None:
            for s in self.surfaces:
                s.draw(ax)

        return figh, ax


# ################################################
# optical surfaces
# ################################################
class surface:
    """
    The geometry of the surface should be defined by the center, input_axis, and output_axis
    """
    def __init__(self, input_axis, output_axis, center, paraxial_center, aperture_rad, is_aperture_stop=False):

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
        self.is_aperture_stop = is_aperture_stop

    def get_normal(self, pts):
        """
        Get normal value at pts. Should be implemented so if pass a ray instead of a point, will
        take the (xo, yo, zo) part and ignore the rest
        """
        pass

    def get_intersect(self, rays, material):
        """
        Find intersection points between rays and surface for rays propagating through a given material,
         and return resulting rays at intersection
        """
        pass

    def propagate(self, ray_array, material1, material2):
        """
        propagate rays throug the surface
        @param ray_array:
        @param material1:
        @param material2:
        @return:
        """
        pass

    def get_seidel_third_order_fns(self):
        raise NotImplementedError()

    def get_ray_transfer_matrix(self, n1, n2):
        pass

    def solve_img_eqn(self, s, n1, n2):
        """
        solve imaging equation where s is the object distance and s' is the image distance

        s and s' use the same convention, where they are negative if to the "left" of the optic and positive
        to the right.

        todo: helper function for calculating seidel aberrations (not finished)
        @param s:
        @param n1:
        @param n2:
        @return:
        """
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

    def is_pt_on_surface(self, pts,):
        """
        Test if a point is on surface
        """
        pass

    def draw(self, ax):
        """
        Draw surface on matplotlib axis
        """
        pass


class refracting_surface(surface):
    def propagate(self, ray_array, material1, material2):
        """
        Given a set of rays, propagate them to the surface of this object and compute their refraction.
        Return the update ray array with these two new ray positions
        :param ray_array: nsurfaces x nrays x 8
        :param material1: material on first side of surface
        :param material2: material on second side of surface
        """
        if ray_array.ndim == 1:
            ray_array = np.expand_dims(ray_array, axis=(0, 1))
        if ray_array.ndim == 2:
            ray_array = np.expand_dims(ray_array, axis=0)

        # get latest rays
        rays = ray_array[-1]
        # find intersection with surface
        rays_intersection = self.get_intersect(rays, material1)
        # compute normals
        normals = self.get_normal(rays_intersection)

        # check ray was coming from the "front side"
        # i.e. the ray has to be coming from the correct 2*pi area of space
        ray_normals = rays[:, 3:6]
        cos_ray_input = np.sum(ray_normals * np.expand_dims(self.input_axis, axis=0), axis=1)
        with np.errstate(invalid="ignore"):
            not_incoming = cos_ray_input < 0
        rays_intersection[not_incoming] = np.nan

        # do refraction
        rays_refracted = refract(rays_intersection, normals, material1, material2)

        # append these rays to full array
        ray_array = np.concatenate((ray_array, np.stack((rays_intersection, rays_refracted), axis=0)), axis=0)

        return ray_array


class reflecting_surface(surface):
    def propagate(self, ray_array, material1, material2=None):
        """
        Given a set of rays, propagate them to the surface of this object and compute their refraction.
        Return the update ray array with these two new ray positions
        :param ray_array: nsurfaces x nrays x 8
        """
        if ray_array.ndim == 1:
            ray_array = np.expand_dims(ray_array, axis=(0, 1))
        if ray_array.ndim == 2:
            ray_array = np.expand_dims(ray_array, axis=0)

        # get latest rays
        rays = ray_array[-1]
        # find intersection with surface
        rays_intersection = self.get_intersect(rays, material1)
        # compute normals
        normals = self.get_normal(rays_intersection)
        # do refraction
        rays_refracted = reflect(rays_intersection, normals)

        # append these rays to full array
        ray_array = np.concatenate((ray_array, np.stack((rays_intersection, rays_refracted), axis=0)), axis=0)

        return ray_array


class flat_surface(refracting_surface):
    """
    Surface is defined by
        [(x, y, z) - (cx, cy, cz)] \cdot normal = 0

    Where the normal should point along the intended direction of ray-travel
    """
    def __init__(self, center, normal, aperture_rad, is_aperture_stop=False):
        self.normal = np.array(normal).squeeze()
        super().__init__(normal, normal, center, center, aperture_rad, is_aperture_stop)


    def get_normal(self, pts):
        pts = np.atleast_2d(pts)
        normal = np.atleast_2d(self.normal)
        return np.tile(normal, (pts.shape[0], 1))


    def get_intersect(self, rays, material):
        """

        """
        rays_int, ts = propagate_ray2plane(rays, self.normal, self.center, material, exclude_backward_propagation=True)

        return rays_int


    def is_pt_on_surface(self, pts):
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


    def draw(self, ax):
        # take Y = 0 portion of surface
        y_hat = np.array([0, 1, 0])
        normal_proj = self.normal - self.normal.dot(y_hat) * y_hat
        normal_proj = normal_proj / np.linalg.norm(normal_proj)

        # plane projected in XZ plane follows this direction
        dv = np.cross(normal_proj, y_hat)

        ts = np.linspace(-self.aperture_rad, self.aperture_rad, 101)

        # construct line using broadcasting
        pts = np.expand_dims(self.center, axis=0) + np.expand_dims(ts, axis=1) * np.expand_dims(dv, axis=0)

        ax.plot(pts[:, 2], pts[:, 0], 'k')


class plane_mirror(reflecting_surface):
    """
    Surface is defined by
        [(x, y, z) - (cx, cy, cz)] \cdot normal = 0

    Where the normal should point along the intended direction of ray-travel
    """

    def __init__(self, center, normal, aperture_rad, is_aperture_stop=False):
        self.normal = np.array(normal).squeeze()
        super().__init__(normal, normal, center, center, aperture_rad, is_aperture_stop)

    def get_normal(self, pts):
        pts = np.atleast_2d(pts)
        normal = np.atleast_2d(self.normal)
        return np.tile(normal, (pts.shape[0], 1))

    def get_intersect(self, rays, material):
        """

        """
        rays_int, ts = propagate_ray2plane(rays, self.normal, self.center, material)
        # exclude rays which will not intersect plane (but would have intersected in the past)
        rays_int[ts < 0] = np.nan

        return rays_int

    def is_pt_on_surface(self, pts):
        pts = np.atleast_2d(pts)
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
        xc, yc, zc = self.center
        nx, ny, nz = self.normal

        on_surface = np.abs((x - xc) * nx + (y - yc) * ny + (z - zc) * nz) < 1e-12

        return on_surface

    def get_ray_transfer_matrix(self, n1, n2):
        mat = np.array([[1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, -1]])
        return mat

    def draw(self, ax):
        # take Y = 0 portion of surface
        y_hat = np.array([0, 1, 0])
        normal_proj = self.normal - self.normal.dot(y_hat) * y_hat
        normal_proj = normal_proj / np.linalg.norm(normal_proj)

        # plane projected in XZ plane follows this direction
        dv = np.cross(normal_proj, y_hat)

        ts = np.linspace(-self.aperture_rad, self.aperture_rad, 101)

        # construct line using broadcasting
        pts = np.expand_dims(self.center, axis=0) + np.expand_dims(ts, axis=1) * np.expand_dims(dv, axis=0)

        ax.plot(pts[:, 2], pts[:, 0], 'k')


class spherical_surface(refracting_surface):
    def __init__(self, radius, center, aperture_rad, input_axis=(0, 0, 1), is_aperture_stop=False):
        """
        Constructor method, but often it is more convenient to use get_on_axis() instead.

        :param radius:
        :param center: [cx, cy, cz]
        """
        self.radius = radius

        paraxial_center = np.array(center).squeeze() - self.radius * np.array(input_axis).squeeze()
        super().__init__(input_axis, input_axis, center, paraxial_center, aperture_rad, is_aperture_stop)

    @classmethod
    def get_on_axis(cls, radius, surface_z_position, aperture_rad, is_aperture_stop=False):
        """
        Construct spherical surface from position on-optical axis, instead of center

        Think of this as an alternate constructor
        """
        input_axis = (0, 0, 1)
        return cls(radius, [0, 0, surface_z_position + radius], aperture_rad, input_axis, is_aperture_stop)


    def get_normal(self, pts):
        """
        Return the outward facing normal if self.aperture_radius > 0, otherwise the inward facing normal

        :param pts: each pt defined by row of matrix
        """
        pts = np.atleast_2d(pts)[:, :3]
        normals = (pts - np.expand_dims(np.array(self.center), axis=0)) / self.radius
        return normals


    def get_intersect(self, rays, material):
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


    def is_pt_on_surface(self, pts):
        """
        Check if point is on sphere surfaces
        """
        pts = np.atleast_2d(pts)
        diff = (pts[:, 0] - self.center[0]) ** 2 + (pts[:, 1] - self.center[1]) ** 2 + (pts[:, 2] - self.center[2]) ** 2
        on_surface = np.abs(diff - self.radius**2) < 1e-12
        return on_surface


    def get_seidel_third_order_fns(self, n1, n2, s, sp, t, tp, h, H):
        if s != 0:
            K = n1 * (1 / self.radius - 1 / s) # abbe invariant for image/obj point
            c1 = (1 / (n2 * sp) - 1 / (n1 * s))
        else: # in this case h = 0, and K only enters with a factor of h so doesn't matter...
            K = 0
            c1 = 0

        if t != 0:
            L = n1 * (1 / self.radius - 1 / t) # abbe invariant for pupil/pupil image point
            c2 = (1 / (n2 * tp) - 1 / (n1 * t))
        else: # in this case H = 0, and L only enters with a factor of H so doesn't matter...
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

    def get_ray_transfer_matrix(self, n1, n2):
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


    def draw(self, ax):
        # todo: modify to allow arbitrary input axis
        theta_max = np.arcsin(self.aperture_rad / np.abs(self.radius))
        thetas = np.linspace(-theta_max, theta_max, 101)
        pts_z = self.center[2] - self.radius * np.cos(thetas)
        pts_x = self.center[0] - self.radius * np.sin(thetas)
        ax.plot(pts_z, pts_x, 'k')


class perfect_lens(refracting_surface):
    def __init__(self, focal_len, center, normal, alpha, is_aperture_stop=False):
        """
        This lens has no length. The center position defines the principal planes.
        The normal should point the same direction rays propagate through the system
        The focal point is focal_len away from the surface

        :param focal_len:
        :param center:
        :param normal:
        :param alpha: maximum angle
        todo: not sure what is best way to handle that in general. Should it be at lens
        """
        self.focal_len = focal_len
        self.alpha = alpha
        self.normal = np.array(normal).squeeze()
        aperture_rad = focal_len * np.sin(self.alpha) # only correct up to factor of n1
        super().__init__(normal, normal, center, center, aperture_rad, is_aperture_stop)

    def get_normal(self):
        pass


    def get_intersect(self, rays, material):
        rays_int, ts = propagate_ray2plane(rays, self.normal, self.center, material)
        with np.errstate(invalid="ignore"):
            rays_int[ts < 0] = np.nan
        return rays_int


    def is_pt_on_surface(self, pts):
        pts = np.atleast_2d(pts)
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
        xc, yc, zc = self.center
        nx, ny, nz = self.normal

        on_surface = np.abs((x - xc) * nx + (y - yc) * ny + (z - zc) * nz) < 1e-12

        return on_surface


    def propagate(self, rays, material1, material2):
        """
        Given a set of rays, propagate them to the surface of this object and compute their refraction.
        Return the update ray array with these two new ray positions

        Construction: consider lens as a plane surface distance f from FFP and BFP. Consider rays
        in the FFP. We know that an on axis ray at angle theta. We know this becomes an on-axis ray
        at distance n*focal len * sin(theta) from the optical axis.

        Suppose we have the Fourier shift theorem perfectly satisfied, then we can use this to infer the angles
        of the other rays, recalling that in the BFP the spatial frequency f = -xp / fl / lambda
        e.g. for 1D case sin(theta_p) = xo/fl

        So the full mapping from FFP to BFP is
        (h, sin(theta) ) -> (n1 * fl * sin(theta), -h / fl / n2)
        And the front and back focal planes are located distance n1*f before and n2*f after the lens respectively

        We can see that cascading two of these lenses together ensures the Abbe sine condition is satisfied

        Note: there will be a positional discontinuity at the surface of the lens. For example, given a beam parallel
        to the optical axis incident at height h, let's compute the height after the lens, determined by the angle
        which the beam focuses. To see this, note
        We known theta = sin^(-1)(x / fl) and the
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

        :param ray_array: nsurfaces x nrays x 8
        :param n1: index of refraction on first side of surface
        :param n2: index of refraction on second side of surface
        """

        if rays.ndim == 1:
            rays = np.expand_dims(rays, axis=(0, 1))
        elif rays.ndim == 2:
            rays = np.expand_dims(rays, axis=0)

        wls = rays[-1, :, 7]

        # #####################################
        # get the three surfaces we will need in our calculation: front focal plane (i.e. before the lens),
        # lens surface, back focal plane (i.e. after the lens)
        # #####################################
        # todo: have to somehow handle the fact these planes depend on the wavelength/index of refraction
        front_focal_pts = np.expand_dims(self.center, axis=0) - np.expand_dims(self.normal, axis=0) * self.focal_len * np.expand_dims(material1.n(wls), axis=1)
        # front_focal_plane = flat_surface(front_focal_pt, self.normal, self.aperture_rad)

        # lens_plane = flat_surface(self.center, self.normal, np.inf)

        back_focal_pts = np.expand_dims(self.center, axis=0) + np.expand_dims(self.normal, axis=0) * self.focal_len * np.expand_dims(material2.n(wls), axis=1)

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
        r1_uvec[to_normalize] = r1_uvec[to_normalize] / np.expand_dims(np.linalg.norm(r1_uvec[to_normalize], axis=1), axis=1)

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
        # exclude points which are outside of the NA of the lens
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
        # output ray array is all the rays that were passed in and two new surface
        # (1) right before the lens and (2) right after
        # #####################################
        rays_out = np.concatenate((rays, np.stack((rays_before_lens, rays_after_lens), axis=0)), axis=0)

        return rays_out


    def get_ray_transfer_matrix(self, n1, n2):
        mat = np.array([[1,    0, 0,    0, 0],
                        [-1/self.focal_len, 1, 0,    0, 0],
                        [0,    0, 1,    0, 0],
                        [0,    0, -1/self.focal_len, 1, 0],
                        [0,    0, 0,    0, 1]])
        return mat

    def draw(self, ax):
        # take Y = 0 portion of surface
        y_hat = np.array([0, 1, 0])
        normal_proj = self.normal - self.normal.dot(y_hat) * y_hat
        normal_proj = normal_proj / np.linalg.norm(normal_proj)

        # plane projected in XZ plane follows this direction
        dv = np.cross(normal_proj, y_hat)

        ts = np.linspace(-self.aperture_rad, self.aperture_rad, 101)

        # construct line using broadcasting
        pts = np.expand_dims(self.center, axis=0) + np.expand_dims(ts, axis=1) * np.expand_dims(dv, axis=0)

        ax.plot(pts[:, 2], pts[:, 0], 'k')

# ################################################
# optical materials
# for information about various materials, see https://refractiveindex.info/ or https://www.schott.com
# abbe number vd = (nd - 1) / (nf - nc)
# vd > 50 = crown glass, otherwise flint glass
# ################################################
class material():

    # helium d-line
    wd = 0.5876
    # hydrogen F-line
    wf = 0.4861
    # hydrogen c-line
    wc = 0.6563

    def __init__(self, b_coeffs, c_coeffs):
        # Sellmeier dispersion formula coefficients
        self.b1, self.b2, self.b3 = np.array(b_coeffs).squeeze()
        self.c1, self.c2, self.c3 = np.array(c_coeffs).squeeze()

        # abbe number (measure of dispersion)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.vd = (self.n(self.wd) - 1) / (self.n(self.wf) - self.n(self.wc))

    def n(self, wavelength):
        """
        compute index of refraction from Sellmeier dispersion formula. To use another method with a specific material,
        override this function the derived class

        see https://www.schott.com/d/advanced_optics/02ffdb0d-00a6-408f-84a5-19de56652849/1.2/tie_29_refractive_index_and_dispersion_eng.pdf
        """
        val = self.b1 * wavelength ** 2 / (wavelength ** 2 - self.c1) + \
              self.b2 * wavelength ** 2 / (wavelength ** 2 - self.c2) + \
              self.b3 * wavelength ** 2 / (wavelength ** 2 - self.c3)
        return np.sqrt(val + 1)


class vacuum(material):
    def __init__(self):
        bs = [0., 0., 0.]
        cs = [0., 0., 0.]
        super(vacuum, self).__init__(bs, cs)


class constant(material):
    def __init__(self, n):
        self._n = float(n)
        self.b1 = None
        self.b2 = None
        self.b3 = None
        self.c1 = None
        self.c2 = None
        self.c3 = None
        self.vd = None

    def n(self, wavelength):
        if isinstance(wavelength, float):
            ns = self._n
        else:
            wavelength = np.atleast_1d(np.array(wavelength))
            ns = np.ones(wavelength.shape) * self._n

        return ns


class fused_silica(material):
    def __init__(self):
        bs = [0.6961663, 0.4079426, 0.8974794]
        cs = [0.0684043**2, 0.1162414**2, 9.896161**2]
        super(fused_silica, self).__init__(bs, cs)


# crown glasses (low dispersion, low refractive index)
class bk7(material):
    def __init__(self):
        bs = [1.03961212, 0.231792344, 1.01046945]
        cs = [0.00600069867, 0.0200179144, 103.560653]
        super(bk7, self).__init__(bs, cs)


class nbak4(material):
    def __init__(self):
        bs = [1.28834642, 0.132817724, 0.945395373]
        cs = [0.00779980626, 0.0315631177, 105.965875]
        super(nbak4, self).__init__(bs, cs)


class nbaf10(material):
    def __init__(self):
        bs = [1.5851495, 0.143559385, 1.08521269]
        cs = [0.00926681282, 0.0424489805, 105.613573]
        super(nbaf10, self).__init__(bs, cs)


class nlak22(material):
    """
    https://www.schott.com/shop/advanced-optics/en/Optical-Glass/N-LAK22/c/glass-N-LAK22
    """
    def __init__(self):
        bs = [1.14229781, 0.535138441, 1.040883850]
        cs = [0.00585778594, 0.0198546147, 100.8340170]
        super(nlak22, self).__init__(bs, cs)


# flint glasses (high dispersion, high refractive index)
class sf10(material):
    def __init__(self):
        bs = [1.62153902, 0.256287842, 1.64447552]
        cs = [0.0122241457, 0.0595736775, 147.468793]
        super(sf10, self).__init__(bs, cs)


class nsf6(material):
    """
    https://www.schott.com/shop/advanced-optics/en/Optical-Glass/N-SF6/c/glass-N-SF6
    """
    def __init__(self):
        bs = [1.77931763, 0.338149866, 2.087344740]
        cs = [0.01337141820, 0.0617533621, 174.0175900]
        super(nsf6, self).__init__(bs, cs)


class sf6(material):
    """
    https://www.schott.com/shop/advanced-optics/en/Optical-Glass/SF6/c/glass-SF6
    """
    def __init__(self):
        bs = [1.72448482, 0.390104889, 1.045728580]
        cs = [0.01348719470, 0.0569318095, 118.5571850]
        super(sf6, self).__init__(bs, cs)


class nsf6ht(material):
    """
    https://www.schott.com/shop/advanced-optics/en/Optical-Glass/N-SF6HT/c/glass-N-SF6HT
    """
    def __init__(self):
        bs = [1.77931763, 0.338149866, 2.087344740]
        cs = [0.01337141820, 0.0617533621, 174.0175900]
        super(nsf6ht, self).__init__(bs, cs)


class sf2(material):
    def __init__(self):
        bs = [1.40301821, 0.231767504, 0.939056586]
        cs = [0.0105795466, 0.0493226978, 112.405955]
        super(sf2, self).__init__(bs, cs)
