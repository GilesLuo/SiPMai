from SiPMai.gen_data.smile_generation import gen_smiles_main
from SiPMai.gen_data.ray_generation import ray_gen_main
from SiPMai.gen_data.prepare_dataset import gen_index_main
import os
import json
from typing import Dict
from SiPMai.utils.dataloader import get_dataset_mean_std


def gen_all_data(smiles_file: str, num_mol: int, csv_file: str, min_atom: int, max_atom: int, num_cpus: int,
                 mol_save_dir: str,
                 resolution: int, blur_sigma: int, use_motion_blur: bool, use_gaussian_noise: bool,
                 gen_original_img: bool, gen_mol_drawing:bool, img_show: bool,
                 train_ratio: float, val_ratio: float, test_ratio: float, split_seed: int,
                 redo_mean_std:bool, mean_std_batch_size:int,
                 development: bool = False) -> None:
    """
    Generate all data required for the molecular images.

    Args:
        smiles_file (str): The path to the file containing SMILES strings.
        num_mol (int): The number of molecules to be generated.
        csv_file (str): The path to the CSV file containing molecular data.
        min_atom (int): The minimum number of atoms for a molecule.
        max_atom (int): The maximum number of atoms for a molecule.
        num_cpus (int): The number of CPUs to be used for data generation.
        mol_save_dir (str): The directory where the generated data should be saved.
        resolution (int): The resolution of the generated images.
        blur_sigma (int): The sigma value for the Gaussian blur applied to the images.
        use_motion_blur (bool): Whether to use motion blur in the images.
        use_gaussian_noise (bool): Whether to add Gaussian noise to the images.
        gen_original_img (bool): Whether to generate original images.
        img_show (bool): Whether to display the generated images.
        train_ratio (float): The ratio of data to be used for training.
        val_ratio (float): The ratio of data to be used for validation.
        test_ratio (float): The ratio of data to be used for testing.
        split_seed (int): The seed for the random splitting of data.
        development (bool, optional): Whether to run in development mode. Defaults to False.

    Raises:
        FileNotFoundError: If the smiles file does not exist.
    """
    if development:
        gen_smiles_main(smiles_file, num_mol, csv_file, min_atom, max_atom)
        print("smiles generated")

    if not os.path.exists(smiles_file):
        raise FileNotFoundError(f"smiles file {smiles_file} not found, please generate smiles first.")
    else:
        with open(smiles_file, 'r') as f:
            smiles_dict: Dict = json.load(f)
            num_mol_ = len(smiles_dict)
            print(f"Found {num_mol_} molecules!")
            if num_mol_ > num_mol:
                print(f"use the first {num_mol} molecules for later steps")
                smiles_dict = {k: smiles_dict[k] for k in list(smiles_dict)[:num_mol]}
    if num_cpus == 0:
        num_cpus = os.cpu_count()

    ray_gen_main(smiles_dict, mol_save_dir,
                 resolution, blur_sigma, use_motion_blur, use_gaussian_noise, gen_original_img, gen_mol_drawing,
                 num_cpus, img_show, development)
    gen_index_main(mol_save_dir, train_ratio, val_ratio, test_ratio, split_seed)
    get_dataset_mean_std(mol_save_dir, redo=redo_mean_std, num_workers=num_cpus, batch_size=mean_std_batch_size
                            )

def main() -> None:
    """
    Main function that parses command line arguments and calls the gen_all function.
    """
    print("Start generating data with preset parameters (100k dataset with 39 <= num_atom <= 200)... ")
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--development", type=bool, default=False,
                        help="Whether to run in development mode. Development mode will run ray locally with serial output."
                             "Defaults to False.")
    # gen smiles args
    parser.add_argument("--smiles_file", type=str, default="pubchem_39_200_100k.json",
                        help="The path to the file containing SMILES strings.")
    parser.add_argument("--num_mol", type=int, default=100000, help="The number of molecules to be generated.")
    parser.add_argument("--csv_file", type=str, default="../CID-SMILES2.csv",
                        help="The path to the CSV file containing molecular data.")
    parser.add_argument("--min_atom", type=int, default=39, help="The minimum number of atoms for a molecule.")
    parser.add_argument("--max_atom", type=int, default=200, help="The maximum number of atoms for a molecule.")

    # gen ray args

    parser.add_argument("--mol_save_dir", type=str, default="pubchem_39_200_100k",
                        help="The directory where the generated data should be saved.")

    parser.add_argument("--resolution", type=int, default=256, help="The resolution of the generated images.")
    parser.add_argument("--blur_sigma", type=int, default=16,
                        help="The sigma value for the Gaussian blur applied to the images.")
    parser.add_argument("--use_motion_blur", type=bool, default=False, help="Whether to use motion blur in the images.")
    parser.add_argument("--use_gaussian_noise", type=bool, default=False,
                        help="Whether to add Gaussian noise to the images.")
    parser.add_argument("--gen_original_img", type=bool, default=True, help="Whether to generate original images.")
    parser.add_argument("--gen_mol_drawing", type=bool, default=True, help="Whether to generate mol images for eye inspection.")
    parser.add_argument("--num_cpus", type=int, default=1, help="The number of CPUs to be used for data generation.")
    parser.add_argument("--show", type=bool, default=False, help="Whether to display the generated images.")

    # data indices args
    parser.add_argument("--split_seed", type=int, default=1, help="The seed for the random splitting of data.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="The ratio of data to be used for training.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="The ratio of data to be used for validation.")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="The ratio of data to be used for testing.")

    # compute mean and std args
    parser.add_argument("--redo", type=bool, default=False, help="Whether to redo the mean and std computation.")
    parser.add_argument("--batch_size", type=int, default=256, help="The batch size to be used for mean and std computation.")
    args = parser.parse_args()

    from pkg_resources import resource_filename
    args.smiles_file = resource_filename('SiPMai', f'smiles/{args.smiles_file}')
    cmd_dir = os.getcwd()
    args.mol_save_dir = os.path.abspath(os.path.join(cmd_dir, args.mol_save_dir))
    print("mol_save_dir set to command execution dir: ", args.mol_save_dir)
    gen_all_data(smiles_file=args.smiles_file, num_mol=args.num_mol, csv_file=args.csv_file, min_atom=args.min_atom,
                 max_atom=args.max_atom, num_cpus=args.num_cpus, mol_save_dir=args.mol_save_dir,
                 resolution=args.resolution, blur_sigma=args.blur_sigma, use_motion_blur=args.use_motion_blur,
                 use_gaussian_noise=args.use_gaussian_noise, gen_original_img=args.gen_original_img, gen_mol_drawing=args.gen_mol_drawing,
                 img_show=False,
                 train_ratio=args.train_ratio, val_ratio=args.val_ratio, test_ratio=args.test_ratio,
                 split_seed=args.split_seed, mean_std_batch_size=args.batch_size, redo_mean_std=args.redo,
                 development=args.development)


if __name__ == "__main__":
    main()
