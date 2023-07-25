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
from SiPMai.utils.img_transform import train_transform, val_transform, test_transform
from torchvision import transforms
from tqdm.auto import tqdm

class MoleculeDataset(Dataset):
    data_dir = "../../pubchem_39_200_100k/"
    # modals = ["img", "graph", "smiles", "instruction"]
    modals = ["img"]
    image_transform = train_transform
    graph_transform = None

    def __init__(self, data_index, **kwargs):
        if not (set(self.modals).issubset({"img", "graph", "smiles", "instruction"}) and self.modals):
            raise ValueError("modals must be a subset of {'img', 'graph', 'smiles', 'instruction'}")
        self._data = data_index
        self._batch_molecule = None

    def batch_Molecule(self):
        if self._batch_molecule is None:  # cache the batch data to avoid repeatedly doing the featurization
            mol_imgs = []
            mol_graphs = []
            mol_adjs = []
            mol_smiles = []

            mol_instructions = []
            labels = []

            for mol_dict in self._data:
                img_path = os.path.join(self.data_dir, mol_dict["img_path"])
                info_path = os.path.join(self.data_dir, mol_dict["info_path"])
                json_path = os.path.join(self.data_dir, mol_dict["json_path"])
                if "img" in self.modals:
                    img = Image.open(img_path).convert('RGB')
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
            self._batch_molecule = [mol_imgs, mol_graphs, mol_adjs, mol_smiles, mol_instructions, labels]  # a list of four lists
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

    def compute_global_mean_std(self, num_workers=0, batch_size=128,):
        n = 0
        s = np.zeros(3)
        s2 = np.zeros(3)
        x = np.empty((512, 512, 3), dtype=np.uint8)
        dataloader = MoleculeDataLoader(self, num_workers=num_workers, batch_size=batch_size, shuffle=False)

        for batch in tqdm((dataloader), desc='Computing mean and std in a running fashion '):
            mol_imgs, _, _, _, _, _ = batch.batch_Molecule()
            x[:] = np.array(mol_imgs) / 255.  # Scale pixel values to [0, 1]
            s += x.sum(axis=(0, 1))
            s2 += np.sum(np.square(x), axis=(0, 1))
            n += x.shape[0] * x.shape[1]

        mean = s / n
        std = np.sqrt((s2 / n) - np.square(mean))

        return mean, std

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


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


def get_dataset_mean_std(data_dir, redo=False, num_workers=0, batch_size=128,):
    # check the mean and std of the dataset, if not exist, calculate them, otherwise load them

    file_path = os.path.join(data_dir, "image_stats.json")
    if os.path.exists(file_path) and not redo:
        print("loading pre-calculated mean and std... for {}".format(data_dir))
        with open(file_path, "r") as f:
            stats = json.load(f)
        mean = stats['mean']
        std = stats['std']
    else:
        dataset = MoleculeDataset(data_dir)
        mean, std = dataset.compute_global_mean_std(num_workers=0, batch_size=128,)
        stats = {'mean': mean.tolist(), 'std': std.tolist()}
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
