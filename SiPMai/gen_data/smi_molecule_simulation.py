import warnings
import os
import json
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Draw
from SiPMai.gen_data.util_fn import uniform_noise, motion_blur, compress_binary_arr, decompress_binary_arr
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
from SiPMai.gen_data.compute_img import cal_atom_projection, bond_location_cal
if not "ENABLE_MAYAVI" in globals():
    try:
        from mayavi import mlab

        ENABLE_MAYAVI = True
    except ImportError as e:
        ENABLE_MAYAVI = False
        warnings.warn("mayavi not found, this will not hinder data generation. "
                      "If img_show is needed, please install mayavi")

def points_height_matrix(smi: str, molecule_name: int, resolution: int, info_dir, json_dir, img_dir,
                         blur_sigma, use_motion_blur, use_gaussian_noise,
                         show=False):
    """
    generate 5 files for each molecule:
    1. img_dir/img.png:                 the image of the molecule with gaussian noise (for simulating electron cloud)
    2. img_dir/orig_img.png:            the image of the molecule without noise, calculated from the 2D coordinates
    3. info_dir/points_info.npz:   -- atom instance binary array, shape: (num_atom, resolution, resolution)
                                         -- bond instance binary array, shape: (num_bond, resolution, resolution)
                                         -- adjacency matrix, shape: (num_atom, num_atom)
                                         -- point height array, shape: (resolution, resolution)
    4. info_dir/mol_drawing.png   a molecule image (only for human inspection)
    5. json_dir/.json                   json file containing the SMILES string and other useful information
    """
    img_name = os.path.join(img_dir, str(molecule_name) + '_img.png')
    orig_img_name = os.path.join(img_dir, str(molecule_name) + '_orig_img.png')
    points_info_name = os.path.join(info_dir, str(molecule_name) + '_points_info.npz')
    drawing_name = os.path.join(info_dir, str(molecule_name) + '_mol_drawing.png')
    json_name = os.path.join(json_dir, str(molecule_name) + '.json')

    # check done
    if os.path.exists(points_info_name) and os.path.exists(img_name) and os.path.exists(
            orig_img_name) and os.path.exists(
            drawing_name) and os.path.exists(json_name):
        print(f"molecule {molecule_name} exists, skip")
        return True

    #         2D molecules were converted into 3D structures,
    #         and mechanical functions were used for Angle correction and optimization
    smile_dict = {}
    mol_2D = Chem.MolFromSmiles(smi)
    smile_dict['smile'] = smi
    mol_2D_H = Chem.AddHs(mol_2D)
    atom_num = mol_2D_H.GetNumAtoms()
    bond_num = mol_2D_H.GetNumBonds()

    moleblock_2D = Chem.MolToMolBlock(mol_2D_H)  # Returns the two-dimensional coordinates of the atoms in the molecule
    if show:
        mol_3D = Chem.MolFromSmiles(smi)
        Draw.ShowMol(mol_3D, size=(550, 550), kekulize=False)
        Draw.ShowMol(mol_2D_H, size=(550, 550), kekulize=False)
    mol_2D_H = Chem.MolFromMolBlock(moleblock_2D, removeHs=False)
    conf = mol_2D_H.GetConformer()

    atom_position = []
    atom_radius = {'C': 0.77, 'H': 0.37, 'O': 0.73, 'S': 1.02, 'N': 0.75, 'P': 1.06, 'Cl': 0.99}
    atom_class = {'0.77': 1, '0.37': 2, '0.73': 3, '1.02': 4, '0.75': 5, '1.06': 6, '0.99': 7}
    for s in range(atom_num):
        x = list(conf.GetAtomPosition(s))
        mol_2D_H.GetAtomWithIdx(s).SetProp('molAtomMapNumber', str(mol_2D_H.GetAtomWithIdx(s).GetIdx()))  # 对原子进行索引
        atom_symbol = mol_2D_H.GetAtoms()[s].GetSymbol()
        if atom_symbol in atom_radius.keys():
            x.append(atom_radius[atom_symbol])
        else:
            raise NotImplementedError("only support C, H, O, S, N, P, Cl")
        x.append(mol_2D_H.GetAtomWithIdx(s).GetIdx())
        atom_position.append(x)
    Draw.MolToFile(mol_2D_H, drawing_name, size=(600, 600))
    # 在三维坐标系绘制得到的原子坐标，可通过原子坐标调整三维坐标轴大小
    if show and ENABLE_MAYAVI:
        plt.figure()  # 得到画面
        ax1 = plt.axes(projection='3d')
        ax1.set_xlim(0, 10)  # X轴，横向向右方向
        ax1.set_ylim(10, 0)  # Y轴,左向与X,Z轴互为垂直
        ax1.set_zlim(0, 10)  # 竖向为Z轴
        # color1 = ['r', 'g', 'b', 'k', 'm']
        # marker1 = ['o', 'v', '1', 's', 'H']
        for x in atom_position:
            ax1.scatter(x[0], x[1], x[2], c='r', marker='o', linewidths=4)
        plt.show()

    points = np.array(atom_position)

    # 计算基平面上的投影的x，y的坐标范围，确定生成的坐标点阵的区域
    x, y, z, r = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
    r_max = max(r)
    x_max, x_min, y_max, y_min = int(max(x) + r_max) + 1, int(min(x) - r_max) - 1, int(max(y) + r_max) + 1, int(
        min(y) - r_max) - 1
    z_min = int(min(z) - r_max) - 1  # 方便后续对基平面进行平移操作

    # Draw a scale model of a 3D molecule (stacked ball model)
    if show and ENABLE_MAYAVI:
        mlab.points3d(x, y, z, r * 2, scale_factor=1, resolution=30, mode="sphere")
        mlab.outline()
        mlab.axes()
        mlab.show()

    # Build the matrix to start recording the height
    points_initial = np.zeros([resolution, resolution])  # initialize
    points_class = np.zeros([resolution, resolution])
    # Initializes the bool matrix that stores the atoms
    points_bool = np.zeros([atom_num, resolution, resolution])
    points_bond_bool = np.zeros([bond_num, resolution, resolution])
    x_axes = np.linspace(x_min, x_max, resolution)
    y_axes = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x_axes, y_axes)
    coordinate_points = np.array([X.ravel(), Y.ravel()])
    coordinate_points_array = coordinate_points.T
    coordinate_points_array = np.array(coordinate_points_array).reshape((resolution, resolution, 2))

    '''
    Start to calculate the height of each atom in the molecule from the base plane. At this time, 
    the base plane is set as the xoy plane, and the virtual projection is made from top to bottom through 
    the point light source cluster, that is, the height of the highest atom contacted is taken as the height recorded
    '''
    points_initial, points_bool, points_class = cal_atom_projection(atom_position, coordinate_points_array,
                                                                    resolution,
                                                                    points_bool,
                                                                    points_initial, points_class, atom_class, z_min)
    points_class = np.where(points_class == 0, 255, points_class)
    # attention! 图像坐标系的原点通常是左上角，而数组的索引是从左到右、从上到下的。因此，在将 NumPy 数组转换为 Pillow 图像对象时，需要将数组进行翻转，使其与图像坐标系保持一致。可以使用以下代码进行翻转：
    im = Image.fromarray(points_class[::-1, :])
    im = im.convert('L')
    im.save(orig_img_name)

    # get blurred image
    points_initial = gaussian_filter(points_initial, sigma=blur_sigma)
    if use_motion_blur:
        points_initial = motion_blur(points_initial)
    if use_gaussian_noise:
        points_initial = uniform_noise(points_initial)
    plt.figure(figsize=(resolution, resolution), dpi=1)
    plt.imshow(points_initial, cmap='gray', origin='lower')
    plt.axis('off')
    fig = plt.gcf()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    fig.savefig(img_name, format='png')
    plt.close()

    # todo: calculate per pixel label of each bond in the image
    bond_list = [[] for i in range(bond_num)]
    molecule_adjacent_matrix = np.zeros((atom_num, atom_num), dtype=np.bool_)
    bond_record_dic = {}
    bonds = mol_2D_H.GetBonds()  # To traverse the key
    for bond_i in range(bond_num):
        bond_list[bond_i].append(bonds[bond_i].GetIdx())
        atom_begin_index = bonds[bond_i].GetBeginAtomIdx()
        atom_end_index = bonds[bond_i].GetEndAtomIdx()
        bond_record_dic[f'bond-{bond_i}'] = str(atom_begin_index) + str(atom_end_index)
        molecule_adjacent_matrix[atom_begin_index][atom_end_index] = 1
        molecule_adjacent_matrix[atom_end_index][atom_begin_index] = 1
        for t in atom_position:
            if t[4] == atom_begin_index or t[4] == atom_end_index:
                bond_list[bond_i].append(t[0])
                bond_list[bond_i].append(t[1])
                bond_list[bond_i].append(t[3])

    points_bond_bool = bond_location_cal(bond_list, coordinate_points_array, points_bond_bool, resolution)

    # save np.array

    arr_atom = np.array(points_bool, dtype=bool)
    arr_bond = np.array(points_bond_bool, dtype=bool)

    arr_atom_compressed = compress_binary_arr(arr_atom)
    arr_bond_compressed = compress_binary_arr(arr_bond)
    adj_matrix_compressed = compress_binary_arr(molecule_adjacent_matrix)

    np.savez_compressed(points_info_name,
                        arr_atom=arr_atom_compressed, arr_bond=arr_bond_compressed, adj_matrix=adj_matrix_compressed,
                        arr_atom_shape=arr_atom.shape, arr_bond_shape=arr_bond.shape,
                        adj_shape=molecule_adjacent_matrix.shape,
                        molecule_points_height=points_initial)

    # turn on debug mode to check compression
    arr_atom_, arr_bond_, adj, points_initial_ = decompress_binary_arr(points_info_name)
    assert np.array_equal(arr_atom, arr_atom_)
    assert np.array_equal(arr_bond, arr_bond_)
    assert np.array_equal(adj, molecule_adjacent_matrix)
    assert np.array_equal(points_initial, points_initial_)

    json_information = [bond_record_dic, smile_dict]
    with open(json_name, 'w', encoding='utf-8') as f:
        json.dump(json_information, f)
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles_file", type=str, default="../data/pubchem_smiles.json")
    parser.add_argument("--save_dir", type=str, default="../data")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--blur_sigma", type=int, default=24)
    parser.add_argument("--use_motion_blur", type=bool, default=False)
    parser.add_argument("--use_gaussian_noise", type=bool, default=False)
    parser.add_argument("--show", type=bool, default=False)
    args = parser.parse_args()

    # start generation
    with open(args.smiles_file, 'r') as f:
        mol_dict = json.load(f)

    # prepare folders
    info_dir = os.path.join(args.save_dir, "info")
    json_dir = os.path.join(args.save_dir, "json")
    img_dir = os.path.join(args.save_dir, "img")
    os.makedirs(info_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    pbar = tqdm(total=len(mol_dict), desc="Generating molecules")
    for mol, smiles in mol_dict.items():
        task_id = points_height_matrix(smiles, mol, args.resolution, info_dir, json_dir, img_dir,
                                       args.blur_sigma, args.use_motion_blur, args.use_gaussian_noise, args.show)
        pbar.update()
