import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Tuple, List, Iterator, Union, Optional, Dict
from random import Random
import threading
import os
from PIL import Image
from SiPMai.utils.img_transform import build_transform
from torchvision import transforms

class MoleculeDataset(Dataset):
    def __init__(self, data_index, data_dir, modals: List[str], image_transform=None, graph_transform=None, **kwargs):
        if not (set(modals).issubset({"img", "graph", "smiles", "instruction"}) and modals):
            raise ValueError("modals must be a subset of {'img', 'graph', 'smiles', 'instruction'}")
        self.modals = modals
        self.data_index = data_index
        self.data_dir = data_dir
        self.cid = list(self.data_index.keys())
        self.img_files = self.get_directory(self.data_dir, "img")
        self.info_files = self.get_directory(self.data_dir, "info")
        self.json_files = self.get_directory(self.data_dir, "json")
        self.image_transform = image_transform
        self.graph_transform = graph_transform

        self._batch_molecule = None

    def batch_Molecule(self):
        if self._batch_molecule is None: # cache the batch data to avoid repeatedly doing the featurization
            mol_graphs = []
            mol_adjs = []
            mol_imgs = []
            mol_instructions = []
            labels = []

            for cid, mol_smile, imgs_id in zip(self.cid, self.json_files, self.img_files):
                if "img" in self.modals:
                    img = Image.open(imgs_id)
                    mol_imgs.append(img)
                else:
                    mol_imgs.append(None)
                if "graph" in self.modals:
                    raise NotImplementedError
                else:
                    mol_imgs.append(None)
                if "instruction" in self.modals:  # provide a random instruction on the bond and atom with label
                    # randomly pick atom_idx and bond_idx on the molecule
                    # atom_idx = np.random.randint(0, len(mol_smile))
                    # bond_idx = np.random.randint(0, len(mol_smile))
                    # self.create_mask(mol_smile, atom_idx, bond_idx)
                    raise NotImplementedError
                else:
                    mol_imgs.append(None)
                if "smiles" in self.modals:
                    # mol_graph = MolGraph(mol_smile)
                    # mol_graphs.append(mol_graph)
                    raise NotImplementedError
                else:
                    mol_imgs.append(None)

            #     mol_graph = MolGraph(mol_smile)
            #     mol_graphs.append(mol_graph)
            #     if self.img_directory == None:
            #         mol_imgs.append(torch.rand(26, 26))
            #     else:
            #         pass
            #     if self.instruction_directory == None:
            #         mol_instructions.append(torch.rand(26, 26))
            #     labels.append(label)
            #
            # self.batch_mol_graph = [BatchMolGraph(mol_graphs)] # The required type of input of molecule model is List[BatchMolGraph]
            self._batch_molecule = [mol_imgs, mol_graphs, mol_adjs, mol_instructions, labels] # a list of four lists
        return self._batch_molecule

    def get_directory(self, root_dir, key):
        if not key in ["img", "json", "info"]:
            raise KeyError
        # img path is the relative path based on the ROOT_DIR, therefore we need root_dir
        return [os.path.join(root_dir, self.data_index[cid][f"{key}_path"]) for cid in self.cid]

    def get_smiles(self, root_dir):
        smiles = []
        for cid in self.cid:
            with open(os.path.join(root_dir, self.data_index[cid]["json_path"]), "r") as f:
                mol_dict = json.load(f)
            smiles.append(mol_dict["smiles"])
        return smiles

    def get_info(self, npz_file, key):
        if not key in ["arr_atom", "arr_bond", 'adj_matrix', 'molecule_points_height']:
            raise KeyError("Wrong key")
        # Loading compressed data and decompressing it
        data = np.load(npz_file)

        if key in ["arr_atom", "arr_bond", "adj_matrix"]:
            if key == "arr_atom":
                k, s = 'arr_atom', 'arr_atom_shape'
                arr, arr_shape = data[k], data[s]
                arr = np.unpackbits(arr)
                arr = arr.reshape(arr_shape)
                return arr
            elif key == "arr_bond":
                k, s = 'arr_bond', 'arr_bond_shape'
                arr, arr_shape = data[k], data[s]
                arr = np.unpackbits(arr)
                arr = arr.reshape(arr_shape)
                return arr
            elif key == "adj_matrix":
                k = 'adj_matrix'
                arr = data[k].astype(int)
                return arr
            else:
                raise NotImplementedError

        elif key == "molecule_points_height":
            return data['molecule_points_height']
        else:
            raise KeyError

    def create_mask(self, cid, atom_idx: Optional[Union[int, List[int]]]=None,
                    bond_idx: Optional[Union[int, List[int]]]=None):
        """
        create a binary mask from the 3D bool array in the info.npz file
        atom_idx and bond_idx cannot be None at the same time.
        intersection of atom and bond mask is allowed.
        :return: a binary mask with the same H, W as the 3D bool array
        """

        if atom_idx is None and bond_idx is None:
            raise ValueError("atom_idx and bond_idx cannot be None at the same time")
        else:
            atom_mask, bond_mask = 0, 0
            if atom_idx is not None:
                arr = self.get_info(self.info_files[cid], "arr_atom")
                atom_mask = arr[atom_idx, :, :]
            if bond_idx is not None:
                arr = self.get_info(self.info_files[cid], "arr_bond")
                bond_mask = arr[bond_idx, :, :]

            return (atom_mask + bond_mask) > 0
    def __len__(self):
        return len(self.cid)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        mol_img, mol_graph, mol_adj, mol_instruction, label = self._batch_molecule[0][idx]
        if self.image_transform:
            mol_img = self.image_transform(mol_img)
        if self.graph_transform:
            raise NotImplementedError
        return mol_img, mol_graph, mol_adj, mol_instruction, label


def construct_molecule_batch(data: List[Tuple]) -> MoleculeDataset:
    # This is the collate function
    data = MoleculeDataset(data)  # Re-initialize a small MoleculeDataset object with the batch of data
    data.batch_Molecule()  # Forces computation of the _batch_Molecule

    return data  # a MoleculeDataset with only a batch size


# def construct_molecule_batch(data: List[Tuple]) -> MoleculeDataset:
#     '''just for test'''
#     data = MoleculeDataset("D:\\project\\pretrain_imgsl\\Molsurge\\pubchem_39_200_100k\\train_set_index.json")
#     # Re-initialize a small MoleculeDataset object with the batch of data
#     data.batch_Molecule()  # Forces computation of the _batch_Molecule
#
#     return data  # a MoleculeDataset with only a batch size


class MoleculeSampler(Sampler):
    """A :class:`MMoleculeSampler` samples data from a :class:`MoleculeDataset` for a :class:`MoleculeDataLoader`."""

    def __init__(self,
                 dataset: MoleculeDataset,
                 shuffle: bool = False,
                 seed: int = 0):
        """
        :param shuffle: Whether to shuffle the data.
        :param seed: Random seed. Only needed if :code:`shuffle` is True.
        """
        super(Sampler, self).__init__()

        self.dataset = dataset
        self.shuffle = shuffle

        self._random = Random(seed)

        self.length = len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        """Creates an iterator over indices to sample."""

        indices = list(range(len(self.dataset)))

        if self.shuffle:
            self._random.shuffle(indices)

        return iter(indices)

    def __len__(self) -> int:
        """Returns the number of indices that will be sampled."""
        return self.length


class MoleculeDataLoader(DataLoader):
    """A :class:`MoleculeDataLoader` is a PyTorch :class:`DataLoader` for loading a :class:`MoleculeDataset`."""

    def __init__(self,
                 dataset: MoleculeDataset,
                 batch_size: int = 4,
                 num_workers: int = 8,
                 shuffle: bool = False,
                 seed: int = 0):
        """
        :param dataset: The :class:`MoleculeDataset` containing the imgs,mols and instructions to load.
        :param batch_size: Batch size.
        :param num_workers: Number of workers used to build batches.
        :param shuffle: Whether to shuffle the data.
        :param seed: Random seed. Only needed if shuffle is True.
        """
        self._dataset = dataset
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._shuffle = shuffle
        self._seed = seed
        self._context = None
        self._timeout = 0
        is_main_thread = threading.current_thread() is threading.main_thread()
        if not is_main_thread and self._num_workers > 0:
            self._context = 'forkserver'  # In order to prevent a hanging
            self._timeout = 3600  # Just for sure that the DataLoader won't hang

        self._sampler = MoleculeSampler(
            dataset=self._dataset,
            shuffle=self._shuffle,
            seed=self._seed
        )

        super(MoleculeDataLoader, self).__init__(
            dataset=self._dataset,
            batch_size=self._batch_size,
            sampler=self._sampler,
            num_workers=self._num_workers,
            collate_fn=construct_molecule_batch,
            multiprocessing_context=self._context,
            timeout=self._timeout
        )

    @property
    def iter_size(self) -> int:
        """Returns the number of data points included in each full iteration through the :class:`MoleculeDataLoader`."""
        return len(self._sampler)

    def __iter__(self) -> Iterator[MoleculeDataset]:
        r"""Creates an iterator which returns :class:`MoleculeDataset`\ s"""
        return super(MoleculeDataLoader, self).__iter__()


if __name__ == '__main__':
    # build transform
    input_size = (224, 224)
    auto_augment = True
    interpolation = transforms.InterpolationMode.BILINEAR
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    horizontal_flip_prob = 0.2
    vertical_flip_prob = 0.2
    rotation_range = 15
    affine_translate = (0.1, 0.1)
    erase_prob = 0.
    train_transform = build_transform("train", input_size, auto_augment, interpolation, mean, std,horizontal_flip_prob,
                                      vertical_flip_prob, rotation_range, translate=affine_translate, erase_prob=erase_prob)

    # Test the data loader
    dataset_json = "../../pubchem_39_200_100k/train_set_index.json"
    with open(dataset_json, "r") as f:
        data_index = json.load(f)
    data_dir =  "../../pubchem_39_200_100k/"
    dataset = MoleculeDataset(data_index, data_dir, transform=train_transform, modals=['img'])

    dataloader = MoleculeDataLoader(dataset, num_workers=0, batch_size=2, shuffle=True)



    print('cool')
    for batch in dataloader:
        mol_imgs, smiles, labels, batch_mol_graph = batch.batch_Molecule()
        print('cool')