import os
import json

import ray
from tqdm import tqdm
import time
from rdkit import Chem
from rdkit.Chem import Draw
from SiPMai.gen_data.util_fn import uniform_noise, motion_blur, compress_binary_arr
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from SiPMai.gen_data.compute_img import cal_atom_projection, bond_location_cal
from scipy.ndimage import gaussian_filter

ENABLE_MAYAVI = False


@ray.remote
def points_height_matrix(smi: str, molecule_name: int, resolution: int, info_dir, json_dir, img_dir,
                         blur_sigma, use_motion_blur, use_gaussian_noise,
                         gen_original_img, gen_mol_drawing,
                         show=False):
    """
    generate 5 files for each molecule:
    1. img_dir/img.png:                 the image of the molecule with gaussian noise (for simulating electron cloud)
    2. img_dir/orig_img.png:            binary image of the molecule (only for human inspection)
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

    if show:
        raise NotImplementedError("show is not implemented for ray version")

    # check done
    if os.path.exists(points_info_name) and os.path.exists(img_name) and os.path.exists(
            orig_img_name) and os.path.exists(
        drawing_name) and os.path.exists(json_name):
        print(f"molecule {molecule_name} exists, skip")
        return True

    #         2D molecules were converted into 3D structures,
    #         and mechanical functions were used for Angle correction and optimization
    mol_2D = Chem.MolFromSmiles(smi)
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

    atom_position = np.zeros((atom_num, 5), dtype=np.float32)
    atom_radius = {'C': 0.77, 'H': 0.37, 'O': 0.73, 'S': 1.02, 'N': 0.75, 'P': 1.06, 'Cl': 0.99}
    for s in range(atom_num):
        mol_2D_H.GetAtomWithIdx(s).SetProp('molAtomMapNumber', str(mol_2D_H.GetAtomWithIdx(s).GetIdx()))
        atom_symbol = mol_2D_H.GetAtoms()[s].GetSymbol()
        # get x, y, z, r
        if atom_symbol in atom_radius.keys():
            atom_position[s, 3] = atom_radius[atom_symbol]
        else:
            raise NotImplementedError("only support C, H, O, S, N, P, Cl")
        atom_position[s, :3] = list(conf.GetAtomPosition(s))
        atom_position[s, 4] = mol_2D_H.GetAtomWithIdx(s).GetIdx()  # atom index

    # save drawing
    if gen_mol_drawing:
        Draw.MolToFile(mol_2D_H, drawing_name, size=(600, 600))

    # get image boundaries
    x, y, z, r = atom_position[:, 0], atom_position[:, 1], atom_position[:, 2], atom_position[:, 3]
    r_max = r.max()
    x_max, x_min, y_max, y_min = int(x.max() + r_max) + 1, int(x.min() - r_max) - 1, int(y.max() + r_max) + 1, int(
        y.min() - r_max) - 1
    z_min = int(z.min() - r_max) - 1  # doesn't really matter

    # get x, y mesh
    points_bond_bool = np.zeros([bond_num, resolution, resolution], dtype=bool)
    x_axes = np.linspace(x_min, x_max, resolution)
    y_axes = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x_axes, y_axes)
    coordinate_points = np.array([X.ravel(), Y.ravel()])
    mesh = coordinate_points.T
    mesh = np.array(mesh).reshape((resolution, resolution, 2))

    '''
    Start to calculate the height of each atom in the molecule from the base plane. At this time, 
    the base plane is set as the xoy plane, and the virtual projection is made from top to bottom through 
    the point light source cluster, that is, the height of the highest atom contacted is taken as the height recorded
    '''
    height_mesh, atom_mask, is_coincide = cal_atom_projection(atom_position, mesh, z_min)
    # if is_coincide.any():
    #     raise ValueError("atom coincide")
    if gen_original_img:
        orig_arr = atom_mask.sum(axis=0)
        orig_arr[orig_arr > 0] = 255
        im = Image.fromarray(orig_arr[::-1, :])
        im = im.convert('L')
        im.save(orig_img_name)

    # get blurred image
    height_mesh = gaussian_filter(height_mesh, sigma=blur_sigma)
    if use_motion_blur:
        height_mesh = motion_blur(height_mesh)
    if use_gaussian_noise:
        height_mesh = uniform_noise(height_mesh)
    plt.figure(figsize=(resolution, resolution), dpi=1)
    plt.imshow(height_mesh, cmap='gray', origin='lower')
    plt.axis('off')
    fig = plt.gcf()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    fig.savefig(img_name, format='png')
    plt.close()

    # get adjacent matrix
    bonds = mol_2D_H.GetBonds()  # To traverse the key
    molecule_adjacent_matrix = np.zeros((atom_num, atom_num), dtype=np.bool_)
    bond_record_dic = {}
    bond_list = [[bond.GetIdx()] for bond in bonds]

    for bond_i, bond in enumerate(bonds):
        atom_begin_index = bond.GetBeginAtomIdx()
        atom_end_index = bond.GetEndAtomIdx()
        bond_key = f"{atom_begin_index}-{atom_end_index}"
        bond_record_dic[bond_key] = bond_i
        molecule_adjacent_matrix[atom_begin_index, atom_end_index] = 1
        molecule_adjacent_matrix[atom_end_index, atom_begin_index] = 1

        matching_positions = atom_position[
            (atom_position[:, 4] == atom_begin_index) | (atom_position[:, 4] == atom_end_index)]
        bond_list[bond_i].extend(matching_positions[:, [0, 1, 3]].ravel().tolist())

    # get bond mask in the image
    points_bond_bool = bond_location_cal(bond_list, mesh, points_bond_bool, resolution)

    # save np.array

    arr_atom = np.array(atom_mask, dtype=bool)
    arr_bond = np.array(points_bond_bool, dtype=bool)

    arr_atom_compressed = compress_binary_arr(arr_atom)
    arr_bond_compressed = compress_binary_arr(arr_bond)
    adj_matrix_compressed = compress_binary_arr(molecule_adjacent_matrix)

    np.savez_compressed(points_info_name,
                        arr_atom=arr_atom_compressed, arr_bond=arr_bond_compressed, adj_matrix=adj_matrix_compressed,
                        arr_atom_shape=arr_atom.shape, arr_bond_shape=arr_bond.shape,
                        adj_shape=molecule_adjacent_matrix.shape,
                        molecule_points_height=height_mesh)
    try:
        with open(json_name, 'w', encoding='utf-8') as f:
            json.dump({"bond_dict": bond_record_dic, "smiles": smi}, f)
    except Exception as e:
        # to avoid the error of json.dump, remove the file if an exception occurs
        os.remove(json_name)
        raise ValueError("something wrong with json.dump. Json file removed to avoid saving broken files")
    return True


def ray_gen_main(mol_dict, save_dir, resolution, blur_sigma, use_motion_blur, use_gaussian_noise, gen_original_img=False, gen_mol_drawing=False,
                 num_cpu=None, show=False, debug_mode=False):
    ray.init(num_cpus=num_cpu, local_mode=debug_mode)

    # prepare folders
    info_dir = os.path.join(save_dir, "info")
    json_dir = os.path.join(save_dir, "json")
    img_dir = os.path.join(save_dir, "img")
    os.makedirs(info_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    tasks = {}
    for mol, smiles in mol_dict.items():
        task_id = points_height_matrix.remote(smiles, mol, resolution, info_dir, json_dir, img_dir,
                                              blur_sigma, use_motion_blur, use_gaussian_noise,
                                              gen_original_img, gen_mol_drawing, show)
        tasks[task_id] = mol

    pbar = tqdm(total=len(tasks), desc="Processing tasks")
    while len(tasks):
        done_ids, _ = ray.wait(list(tasks.keys()), num_returns=1)
        for ready_id in done_ids:
            try:
                result = ray.get(ready_id)
                molecule_name = tasks.pop(ready_id)
                pbar.set_description(molecule_name)
                pbar.update()
            except ray.exceptions.RayTaskError as ex:
                print(f"Task failed for molecule {tasks[ready_id]} with error: {ex}")
        time.sleep(0.5)  # To avoid busy waiting

    pbar.close()
    ray.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles_file", type=str, default="../smiles/pubchem_39_200_100k.json")
    parser.add_argument("--save_dir", type=str, default="../data")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--blur_sigma", type=int, default=24)
    parser.add_argument("--use_motion_blur", type=bool, default=False)
    parser.add_argument("--use_gaussian_noise", type=bool, default=False)
    parser.add_argument("--show", type=bool, default=False)
    args = parser.parse_args()
    with open(args.smiles_file, 'r') as f:
        mol_dict = json.load(f)
    # start generation
    ray_gen_main(mol_dict, args.save_dir,
                 args.resolution, args.blur_sigma, args.use_motion_blur, args.use_gaussian_noise, args.show, num_cpu=2)
