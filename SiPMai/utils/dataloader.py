import json
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Tuple, List, Iterator
from random import Random
import threading
import os

class MoleculeDataset(Dataset):
    def __init__(self, dataset_json, transform=None, **kwargs):
        with open(dataset_json, "r") as f:
            self.data_index = json.load(f)
        self.data_dir = "/".join(dataset_json.split("/")[:-1])  # assume dataset json is at the root data dir
        self.cid = list(self.data_index.keys())
        self.img_files = self.get_directory(self.data_dir, "img")
        self.info_files = self.get_directory(self.data_dir, "info")
        self.json_files = self.get_directory(self.data_dir, "json")

        self._batch_Molecule = None

    def batch_Molecule(self):
        raise NotImplementedError

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
            elif key == "arr_bond":
                k, s = 'arr_bond', 'arr_bond_shape'
            elif key == "adj_matrix":
                k, s = 'adj_matrix', 'adj_shape'
            else:
                raise NotImplementedError
            arr, arr_shape = data[k], data[s]
            arr = np.unpackbits(arr)
            arr = arr.reshape(arr_shape)
            return arr
        elif key == "molecule_points_height":
            return data['molecule_points_height']
        else:
            raise KeyError

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):

        return self._data[idx]  # a tuple with four elements



def construct_molecule_batch(data: List[Tuple]) -> MoleculeDataset:
    # This is the collate function
    data = MoleculeDataset(data)  # Re-initialize a small MoleculeDataset object with the batch of data
    data.batch_Molecule()  # Forces computation of the _batch_Molecule

    return data  # a MoleculeDataset with only a batch size


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
                 batch_size: int = 50,
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
    # Test the data loader

    dataset = MoleculeDataset("../pubchem_39_200_100k/train_set_index.json")
    # dataloader = MoleculeDataLoader(dataset)
    # print('cool')
    # for batch in dataloader:
    #     mol_imgs, batch_mol_graph, mol_instructions, labels = batch.batch_Molecule()
    #     print(batch_mol_graph[0].a_scope)
    #     print('cool')
