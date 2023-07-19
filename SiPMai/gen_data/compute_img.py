from matplotlib.path import Path
import numpy as np
import math
from numba import jit, njit


@njit
def cal_atom_projection(atom_position, coordinate_points_array, resolution, points_bool, points_initial, points_class,
                        atom_class, z_min):
    def cal_per_atom_position(x1, y1, atom_position, z_min):
        height, idex_list, atom_list = [], [], []
        for a in atom_position:
            x2, y2, r2, idex = a[0], a[1], a[3], a[4]
            z2 = a[2] + abs(z_min)
            length = math.sqrt(math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2))
            if length <= r2:
                point_height = math.sqrt(r2 * r2 - length * length) + z2
                height.append(point_height)
                atom_list.append(r2)
                idex_list.append(idex)
        return np.array(height), np.array(idex_list), np.array(atom_list)

    for r in range(resolution):
        for t in range(resolution):
            x1 = coordinate_points_array[r, t, 0]
            y1 = coordinate_points_array[r, t, 1]
            height, idex_list, atom_list = cal_per_atom_position(x1, y1, atom_position, z_min)
            if len(height) == 0:
                continue
            else:
                point_height_max = np.max(height)
                point_height_max_index = np.argmax(height)
                points_initial[r][t] = point_height_max
                point_idex = idex_list[point_height_max_index]
                points_bool[point_idex, r, t] = 1
                points_class[r, t] = atom_class[str(atom_list[point_height_max_index])]
    return points_initial, points_bool, points_class


def bond_location_cal(bond_list, coordinate_points_array, points_bond_bool, resolution):
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
                x_1 = coordinate_points_array[r, t, 0]
                y_1 = coordinate_points_array[r, t, 1]
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
