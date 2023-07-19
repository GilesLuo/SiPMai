from unittest import TestCase
from ..gen_data.ray_generation import ray_gen_main
from ..gen_data.prepare_dataset import gen_index_main
import os
from pkg_resources import resource_filename
import json


class TestDataGeneration(TestCase):

    def test_smiles_exist(self):
        # check file exists in SiPMai/smiles/ folder
        smiles_dir = resource_filename('SiPMai', 'smiles')
        self.assertTrue(os.path.exists(smiles_dir))

    def test_ray_gen_main(self):
        def take(n, iterable):
            """Return the first n items of the iterable as a dict."""
            return dict(list(iterable)[:n])

        smiles_json = resource_filename('SiPMai', 'smiles/pubchem_39_200_100k.json')
        with open(smiles_json, 'r') as f:
            mol_dict = json.load(f)

        n = 10
        mol_dict = take(n, mol_dict.items())
        cmd_dir = os.getcwd()
        mol_save_dir = os.path.join(cmd_dir, "test_gen_data")
        ray_gen_main(mol_dict, mol_save_dir, 128, 24, False, False, False, True)

        # check if mol_save_dir exists
        self.assertTrue(os.path.exists(mol_save_dir))

        img_dir = os.path.join(mol_save_dir, "img")
        json_dir = os.path.join(mol_save_dir, "json")
        info_dir = os.path.join(mol_save_dir, "info")
        # check if mol_save_dir contains n images
        self.assertEqual(len([f for f in os.listdir(img_dir) if "_img" in f and "orig_img" not in f]), n)
        self.assertEqual(len([f for f in os.listdir(img_dir) if "orig_img" in f]), n)
        # check if mol_save_dir contains n json files
        self.assertEqual(len([f for f in os.listdir(json_dir) if f.endswith('.json')]), n)
        # check if mol_save_dir contains n npz files
        self.assertEqual(len([f for f in os.listdir(info_dir) if f.endswith('.npz')]), n)
        self.assertEqual(len([f for f in os.listdir(info_dir) if f.endswith('.png') and "_mol_drawing.png" in f]), n)

    def test_gen_index_main(self):
        cmd_dir = os.getcwd()
        mol_save_dir = os.path.join(cmd_dir, "test_gen_data")
        gen_index_main(mol_save_dir, 0.6, 0.2, 0.2, 1)

        # check if mol_save_dir contains train, val, test folders
        self.assertTrue(os.path.exists(os.path.join(mol_save_dir, "train_set_index.json")))
        self.assertTrue(os.path.exists(os.path.join(mol_save_dir, "val_set_index.json")))
        self.assertTrue(os.path.exists(os.path.join(mol_save_dir, "test_set_index.json")))

if __name__ == '__main__':
    TestCase
