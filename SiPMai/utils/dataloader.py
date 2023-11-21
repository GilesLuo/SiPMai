import json
import pandas as pd
import numpy as np
import torch
from SiPMai.utils.img_transform import get_dummy_transform, get_dummy_paired_transform
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Tuple, List, Iterator, Union, Optional, Dict, Type
from random import Random
import threading
import os
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm


class MoleculeDataset(Dataset):
    modals = ["img", "graph", "smiles", "instruction"]
    image_transform = get_dummy_paired_transform()
    graph_transform = None

    def __init__(self, data_index, **kwargs):
        self._data = data_index
        self._batch_molecule = None

    def batch_Molecule(self):
        if not (set(self.modals).issubset({"img", "graph", "smiles", "instruction"}) or not self.modals):
            raise ValueError("modals must be a subset of {'img', 'graph', 'smiles', 'instruction'}")

        if self._batch_molecule is None:  # cache the batch data to avoid repeatedly doing the featurization
            mol_imgs = []
            mol_graphs = []
            mol_adjs = []
            mol_smiles = []

            mol_instructions = []
            labels = []

            for mol_dict in self._data:
                img_path = mol_dict["img_path"]
                info_path = mol_dict["info_path"]
                json_path = mol_dict["json_path"]
                if "img" in self.modals:
                    img = Image.open(img_path).convert('RGB')
                    if self.image_transform is not None:
                        img = self.image_transform(img)
                    mol_imgs.append(img)
                else:
                    mol_imgs.append(None)
                if "graph" in self.modals:
                    raise NotImplementedError
                else:
                    mol_graphs.append(None)
                if "instruction" in self.modals:  # provide a random instruction on the bond and atom with label
                    # randomly pick atom_idx and bond_idx on the molecule
                    # atom_idx = np.random.randint(0, len(mol_smile))
                    # bond_idx = np.random.randint(0, len(mol_smile))
                    # self.create_mask(mol_smile, atom_idx, bond_idx)
                    raise NotImplementedError
                else:
                    mol_instructions.append(None)
                if "smiles" in self.modals:
                    # mol_graph = MolGraph(mol_smile)
                    # mol_graphs.append(mol_graph)
                    raise NotImplementedError
                else:
                    mol_smiles.append(None)

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
            self._batch_molecule = [torch.stack(mol_imgs), mol_graphs, mol_adjs, mol_smiles, mol_instructions,
                                    labels]  # a list of four lists
        return self._batch_molecule

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

    def create_mask(self, cid, atom_idx: Optional[Union[int, List[int]]] = None,
                    bond_idx: Optional[Union[int, List[int]]] = None):
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
                arr = self.get_info(self._data[cid], "arr_atom")
                atom_mask = arr[atom_idx, :, :]
            if bond_idx is not None:
                arr = self.get_info(self._data[cid], "arr_bond")
                bond_mask = arr[bond_idx, :, :]

            return (atom_mask + bond_mask) > 0

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


class MeanStdDataset(MoleculeDataset):
    modals = ["img"]
    image_transform = get_dummy_transform()


def wrap_collate_fn(dataset):
    def construct_molecule_batch(data: List[Tuple]) -> MoleculeDataset:
        dataset_class = dataset.__class__
        # This is the collate function
        data = dataset_class(data)  # Re-initialize a small MoleculeDataset object with the batch of data
        data.batch_Molecule()  # Forces computation of the _batch_Molecule

        return data  # a MoleculeDataset with only a batch size

    return construct_molecule_batch


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
                 seed: int = 0, **kwargs):
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

        if "sampler" not in kwargs:
            self._sampler = MoleculeSampler(
                dataset=self._dataset,
                shuffle=self._shuffle,
                seed=self._seed
            )
        else:
            self._sampler = kwargs["sampler"]
            del kwargs["sample"]

        super(MoleculeDataLoader, self).__init__(
            dataset=self._dataset,
            batch_size=self._batch_size,
            sampler=self._sampler,
            num_workers=self._num_workers,
            collate_fn=wrap_collate_fn(self._dataset),
            multiprocessing_context=self._context,
            timeout=self._timeout,
            **kwargs
        )

    @property
    def iter_size(self) -> int:
        """Returns the number of data points included in each full iteration through the :class:`MoleculeDataLoader`."""
        return len(self._sampler)

    def __iter__(self) -> Iterator[MoleculeDataset]:
        r"""Creates an iterator which returns :class:`MoleculeDataset`\ s"""
        return super(MoleculeDataLoader, self).__iter__()


def create_dataset(data_dir, split, dataset_class: Type[MoleculeDataset], image_transform) -> MoleculeDataset:
    if split == "train":
        file_name = "train_set_index.json"
    elif split == "val":
        file_name = "val_set_index.json"
    elif split == "test":
        file_name = "test_set_index.json"
    else:
        raise ValueError("split must be one of ['train', 'val', 'test']")
    data_json = os.path.join(data_dir, file_name)
    with open(data_json, "r") as f:
        data_index = list(json.load(f).values())
    for i in range(len(data_index)):
        for key in data_index[i].keys():
            data_index[i][key] = os.path.join(data_dir, data_index[i][key])
    dataset_class.image_transform = image_transform
    dataset = dataset_class(data_index)
    return dataset


def get_dataset_mean_std(data_dir, redo=False, num_workers=0, batch_size=128, mask_zero=False,
                         device="cuda" if torch.cuda.is_available() else "cpu"):
    # check the mean and std of the dataset, if not exist, calculate them, otherwise load them

    file_path = os.path.join(data_dir, "image_stats.json")

    if os.path.exists(file_path) and not redo:
        print("loading pre-calculated mean and std... for {}".format(data_dir))
        with open(file_path, "r") as f:
            stats = json.load(f)
        mean = stats['mean']
        std = stats['std']
    else:
        dataset = create_dataset(data_dir, split="train", dataset_class=MeanStdDataset,
                                 image_transform=get_dummy_transform())
        dataloader = MoleculeDataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False)

        n = 0
        s = torch.zeros(3).to(device)
        s2 = torch.zeros(3).to(device)
        for batch in tqdm((dataloader), desc='Computing mean and std in a running fashion '):
            mol_imgs, _, _, _, _, _ = batch.batch_Molecule()
            mol_imgs = mol_imgs.to(device)

            if mask_zero:
                mask = (mol_imgs != 0)
                n += mask.sum(axis=(0, 2, 3))
                s += (mol_imgs * mask).sum(axis=(0, 2, 3))
                s2 += ((mol_imgs ** 2) * mask).sum(axis=(0, 2, 3))
            else:
                s += mol_imgs.sum(axis=(0, 2, 3))
                s2 += (mol_imgs ** 2).sum(axis=(0, 2, 3))
                n += mol_imgs.shape[0] * mol_imgs.shape[2] * mol_imgs.shape[3]

        mean = s / n
        std = torch.sqrt((s2 / n) - torch.square(mean))
        stats = {'mean': mean.cpu().numpy().tolist(), 'std': std.cpu().numpy().tolist()}
        with open(file_path, "w") as f:
            json.dump(stats, f)

    return mean, std


if __name__ == '__main__':

    # Test the data loader
    dataset_json = "../../pubchem_39_200_100k/train_set_index.json"
    with open(dataset_json, "r") as f:
        data_index = json.load(f)

    dataset = MoleculeDataset(list(data_index.values()))

    dataloader = MoleculeDataLoader(dataset, num_workers=0, batch_size=2, shuffle=True)

    print('cool')
    for batch in dataloader:
        mol_imgs, graph, adj, smiles, instruction, labels = batch.batch_Molecule()
        print('cool')
