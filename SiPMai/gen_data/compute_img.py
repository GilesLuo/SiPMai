from matplotlib.path import Path
import numpy as np
import math
from numba import jit, njit
from numba.typed import List


def cal_atom_projection(atom_position, mesh, z_min, get_coincidence=False):
    """
    :param atom_position: n_atom (x, y, z, r, ) of the atom
    :param mesh: [H, W, 2] mesh grid of x, y axes
    :param z_min: the minimum z coordinate of the atom
    :return: height [n_atom, H, W], atom mask [n_atom, H, W], bool [H, W], the 1st dim order is the order of atom_position
    """
    def ball_3d(x0, y0, z0, r, x, y):
        # Compute squared distance from the sphere center to the point in the xy-plane
        sq_dist_xy = (x - x0) ** 2 + (y - y0) ** 2
        sq_dist_xy[sq_dist_xy > r ** 2] = r ** 2
        # Calculate z-coordinate (height) of the point on the sphere surface
        z = z0 + np.sqrt(r ** 2 - sq_dist_xy)

        return z

    def compute_height_matrix(xyzr, mesh):
        # xyzr shape: [n, 4] (x0, y0, z0, r for each sphere)
        # mesh shape: [H, W, 2] (x, y coordinate for each point in the mesh)


        n_spheres = xyzr.shape[0]
        H, W = mesh.shape[0], mesh.shape[1]

        height_matrix = np.zeros((n_spheres, H, W))

        for i in range(n_spheres):
            x0, y0, z0, r = xyzr[i]

            # Compute the height for the sphere and store it in the height matrix
            height_matrix[i] = ball_3d(x0, y0, z0, r, mesh[:, :, 0], mesh[:, :, 1])
        return height_matrix

    def check_coincidence(height_matrix):
        # Compute maximum and minimum height for each point in the mesh across all spheres
        pos_mask, neg_mask = height_matrix.copy(), height_matrix.copy()
        pos_mask[pos_mask > 0] = 1
        pos_mask = pos_mask.sum(axis=0)

        neg_mask = neg_mask.sum(axis=0)
        neg_mask[neg_mask > 0] = 1

        coincide_matrix = (pos_mask - neg_mask).astype(bool)
        return coincide_matrix


    xyzr, atom_class = atom_position[:, :4], atom_position[:, 4]
    height_all_atom = compute_height_matrix(xyzr, mesh)
    height = np.max(height_all_atom, axis=0)

    is_coincide = check_coincidence(height_all_atom) if get_coincidence else None
    atom_mask = height_all_atom > 0
    return height + z_min, atom_mask, is_coincide


def bond_location_cal(bond_list, mesh, points_bond_bool, resolution):
    @jit(nopython=True)
    def cal_bond_atom_loc(point1_x, point1_y, point2_x, point2_y, a1_radius, a2_radius):
        bond_height = math.sqrt(math.pow((point2_x - point1_x), 2) + math.pow((point2_y - point2_x), 2)) / 3
        bond_x = (point1_x + point2_x) / 2
        bond_y = (point1_y + point2_y) / 2
        bond_width = (a1_radius + a2_radius) / 6
        dx = point2_x - point1_x
        dy = point2_y - point1_y
        angle = math.atan2(dy, dx)
        return bond_y, bond_x, angle, bond_height, bond_width

    for b in bond_list:
        point1_x, point1_y = b[1], b[2]
        point2_x, point2_y = b[4], b[5]
        a1_radius, a2_radius = b[3], b[6]
        bond_y, bond_x, angle, bond_height, bond_width = cal_bond_atom_loc(point1_x, point1_y, point2_x, point2_y,
                                                                           a1_radius, a2_radius)
        # Determine the bond area
        bond_loc = rect_loc(bond_y=bond_y, bond_x=bond_x, bond_angle=angle, bond_height=bond_height,
                            bond_width=bond_width)
        bond_area = Path(bond_loc)
        for r in range(resolution):
            for t in range(resolution):
                x_1 = mesh[r, t, 0]
                y_1 = mesh[r, t, 1]
                points_bond_bool[b[0], r, t] = bond_area.contains_point((x_1, y_1))
    return points_bond_bool


@jit(nopython=True)
def rect_loc(bond_y, bond_x, bond_angle, bond_height, bond_width):
    """
    @param bond_y: The center of the bond y value
    @param bond_x: The center of the bond x value
    @param bond_angle: The slope of the line between the atom and the origin
    @param bond_height: Bond length
    @param bond_width: Bond width
    @return: Bond area
    """
    xo = np.cos(bond_angle)
    yo = np.sin(bond_angle)
    y1 = bond_y + bond_height / 2 * yo
    x1 = bond_x + bond_height / 2 * xo
    y2 = bond_y - bond_height / 2 * yo
    x2 = bond_x - bond_height / 2 * xo

    return np.array(
        [
            [x1 + bond_width / 2 * yo, y1 - bond_width / 2 * xo],
            [x2 + bond_width / 2 * yo, y2 - bond_width / 2 * xo],
            [x2 - bond_width / 2 * yo, y2 + bond_width / 2 * xo],
            [x1 - bond_width / 2 * yo, y1 + bond_width / 2 * xo],
        ]
    )
