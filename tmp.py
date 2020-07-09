import os
import os.path as osp
import pandas
import numpy as np
import torch


def main():

    print(os.getcwd())
    file = osp.join(os.curdir, 'data', 'cit-HepPh', 'cit-HepPh', 'raw', 'cit-HepPh.txt')
    skiprows = 4

    edge_index = pandas.read_csv(file, sep='\t', header=None, skiprows=skiprows, dtype=np.int64)

    edge_index = torch.from_numpy(edge_index.values).t()
    num_nodes = 34546


if __name__ == '__main__':
    main()
