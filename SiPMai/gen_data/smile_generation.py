import argparse
import json
import os
import csv

from rdkit import Chem
from tqdm.auto import tqdm
from typing import Set

def contains_element(smile: str) -> bool:
    """
    Check if a SMILES string contains only specific atomic symbols.

    Args:
        smile (str): A SMILES string representing a molecule.

    Returns:
        bool: True if the molecule only contains the atomic symbols ['C', 'H', 'O', 'S', 'N', 'P', 'Cl'], False otherwise.
    """
    mol = Chem.MolFromSmiles(smile)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    unique_atoms = set(atoms)
    supply_element = ['C', 'H', 'O', 'S', 'N', 'P', 'Cl']
    valid_symbols = set(supply_element)
    return unique_atoms.issubset(valid_symbols)


def gen_smiles_main(save_name: str, num_mol: int, csv_file: str, min_atom_count: int = 39, max_atom_count: int = 120) -> None:
    """
    Main function to generate molecules and save them in a json file.

    Args:
        save_name (str): The path of the json file to save the molecules.
        num_mol (int): The number of molecules to be generated.
        csv_file (str): The path to the CSV file containing molecular data.
        min_atom_count (int, optional): The minimum number of atoms for a molecule. Defaults to 39.
        max_atom_count (int, optional): The maximum number of atoms for a molecule. Defaults to 120.
    """
    # If results file doesn't exist, initialize an empty dictionary.
    # Otherwise, load existing results.
    len_dict = 0
    smiles_dict = {}
    if os.path.isfile(save_name):
        with open(save_name, 'r') as f:
            smiles_dict = json.load(f)
        if len(smiles_dict) < num_mol:
            print(f"Only {len(smiles_dict)} molecules found, generating more...")
            len_dict = len(smiles_dict)
        else:
            print(f"Found {len(smiles_dict)} molecules, skipping generation...")
            return

    pbar = tqdm(total=num_mol - len_dict, desc="Generating molecules", leave=False)
    i = 0
    with open(csv_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)  # Skip header
        for row in tqdm(reader):  # tqdm adds a progress bar
            try:
                cid = row[0]
                smile = row[1]
                atom_count = int(row[2])
                if smile not in smiles_dict.keys():
                    if max_atom_count >= atom_count >= min_atom_count and contains_element(smile):
                        smiles_dict[f"cid{cid}"] = smile
                        i += 1
                        pbar.update(1)
                if i >= num_mol:
                    break
            except ValueError:
                continue

    # Save the dictionary
    with open(save_name, 'w') as f:
        json.dump(smiles_dict, f)
