import os
import os.path as osp
from collections import OrderedDict

import torch
import pandas
import numpy as np
from torch_sparse import coalesce
from torch_geometric.data import (Data, InMemoryDataset, download_url, extract_gz, extract_tar)
from torch_geometric.data.makedirs import makedirs


def read_cit(files, name):
    skiprows = 4

    edge_index = pandas.read_csv(files[0], sep='\t', header=None, skiprows=skiprows, dtype=np.int64)

    edge_index = edge_index.values
    index_dict = OrderedDict()
    k = 0
    for i in range(edge_index.shape[0]):
        for j in range(edge_index.shape[1]):
            id = edge_index[i][j]
            if not id in index_dict.keys():
                index_dict[id] = k
                k += 1
            edge_index[i][j] = index_dict[id]

    edge_index = torch.from_numpy(edge_index).t()
    num_nodes = len(index_dict)
    edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)

    return [Data(edge_index=edge_index, num_nodes=num_nodes)], index_dict


def count_cit(files, name):
    skiprows = 1

    dates = pandas.read_csv(files[1], sep='\t', header=None, skiprows=skiprows, dtype=str)






class HepPhDataset(InMemoryDataset):
    r"""A variety of graph datasets collected from `SNAP at Stanford University
    <https://snap.stanford.edu/data>`_.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    url = 'https://snap.stanford.edu/data'

    available_datasets = {
        'cit-hepph': ['cit-HepPh.txt.gz', 'cit-HepPh-dates.txt.gz'],
    }

    def __init__(self, root, name='cit-hepph', transform=None, pre_transform=None, pre_filter=None):
        self.name = name.lower()
        assert self.name in self.available_datasets.keys()
        super(HepPhDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self):
        return 'data.pt'

    def _download(self):
        if osp.isdir(self.raw_dir) and len(os.listdir(self.raw_dir)) > 0:
            return

        makedirs(self.raw_dir)
        self.download()

    def download(self):
        for name in self.available_datasets[self.name]:
            path = download_url('{}/{}'.format(self.url, name), self.raw_dir)
            print(path)
            if name.endswith('.tar.gz'):
                extract_tar(path, self.raw_dir)
            elif name.endswith('.gz'):
                extract_gz(path, self.raw_dir)
            os.unlink(path)

    def process(self):
        raw_dir = self.raw_dir
        filenames = os.listdir(self.raw_dir)
        if len(filenames) == 1 and osp.isdir(osp.join(raw_dir, filenames[0])):
            raw_dir = osp.join(raw_dir, filenames[0])

        raw_files = sorted([osp.join(raw_dir, f) for f in os.listdir(raw_dir)])

        if self.name[:4] == 'cit-':
            data_list, index_dict = read_cit(raw_files, self.name[4:])
        else:
            raise NotImplementedError

        if len(data_list) > 1 and self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]





        torch.save(self.collate(data_list), self.processed_paths[0])

    def __repr__(self):
        return 'SNAP-{}({})'.format(self.name, len(self))