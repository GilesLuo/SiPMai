import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
import argparse
from tqdm import tqdm

def generate_index_dict(ROOT_DIR):
    IMG_DIR, INFO_DIR, JSON_DIR = os.path.join(ROOT_DIR, 'img'), os.path.join(ROOT_DIR, 'info'), os.path.join(ROOT_DIR, 'json')
    # load all file names in the directory
    img_names = [name for name in os.listdir(IMG_DIR)
                 if os.path.isfile(os.path.join(IMG_DIR, name)) and "orig_img" not in name]
    info_names = [name for name in os.listdir(INFO_DIR)
                  if os.path.isfile(os.path.join(INFO_DIR, name)) and name.endswith("npz")]
    json_names = [name for name in os.listdir(JSON_DIR)
                  if os.path.isfile(os.path.join(JSON_DIR, name))]

    # check if the number of files are the same
    if not len(img_names) == len(info_names) == len(json_names):
        warnings.warn("The number of files in the directories are not the same! Rerun mol generation")

    molecule_dict = {}
    for img, info, json_file in tqdm(zip(sorted(img_names), sorted(info_names), sorted(json_names)),
                                     desc="Generating index dict"):
        # extract molecule id from the file name
        mol_id = "_".join(img.split('_')[:-1])
        if os.path.exists(os.path.join(IMG_DIR, img)) \
                and os.path.exists(os.path.join(INFO_DIR, info))\
                and os.path.exists(os.path.join(JSON_DIR, json_file)):
            molecule_dict[mol_id] = {
                "img_path": os.path.join('img', img),  # under the ROOT_DIR to avoid relative path problem
                "info_path": os.path.join('info', info),
                "json_path": os.path.join('json', json_file)
            }

    return molecule_dict

def gen_index_main(ROOT_DIR, train_ratio, val_ratio, test_ratio, seed):
    mol_dict = generate_index_dict(ROOT_DIR)
    # List all the molecules
    # Set the splitting ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1) < 1e-5, "The sum of ratios should be 1"
    # Split the molecules
    train, test = train_test_split(list(mol_dict.keys()), test_size=1 - train_ratio, random_state=seed)
    val, test = train_test_split(test, test_size=test_ratio / (test_ratio + val_ratio), random_state=seed)

    # Generate dictionaries
    train_dict = {mol: mol_dict[mol] for mol in train}
    val_dict = {mol: mol_dict[mol] for mol in val}
    test_dict = {mol: mol_dict[mol] for mol in test}

    # save dictionaries as json files
    with open(os.path.join(ROOT_DIR, 'train_set_index.json'), 'w') as fp:
        json.dump(train_dict, fp)

    with open(os.path.join(ROOT_DIR, 'val_set_index.json'), 'w') as fp:
        json.dump(val_dict, fp)

    with open(os.path.join(ROOT_DIR, 'test_set_index.json'), 'w') as fp:
        json.dump(test_dict, fp)
    print("dataset indices generated!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example of argparse usage.')
    parser.add_argument("--ROOT_DIR", type=str, default="../data")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    args = parser.parse_args()
    gen_index_main(args.ROOT_DIR, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
