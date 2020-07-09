import os.path as osp

import torch
from torch_geometric.datasets import Planetoid
from HepPh import HepPhDataset
import torch_geometric.transforms as T
from torch_geometric.nn import Node2Vec

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def test(model, data):
    model.eval()
    z = model()
    acc = model.test(z[data.train_mask], data.y[data.train_mask],
                     z[data.test_mask], data.y[data.test_mask], max_iter=150)
    return acc


def main():

    dim = 8
    # dataset = 'Cora'
    dataset = 'cit-HepPh'

    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', dataset)
    # dataset = Planetoid(path, dataset)
    dataset = HepPhDataset(path, dataset, transform=T.TargetIndegree())

    data = dataset[0]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Node2Vec(data.edge_index, embedding_dim=dim, walk_length=20,
                     context_size=10, walks_per_node=10, num_negative_samples=1,
                     sparse=True).to(device)

    loader = model.loader(batch_size=128, shuffle=True, num_workers=2)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)

    for epoch in range(1, 101):
        loss = train(model, loader, optimizer, device)
        # acc = test(model, data)
        acc = 0
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {acc:.4f}')

    torch.save(model.embedding.weight.cpu().detach(), 'embed_{}.pt'.format(dim))


def plot_points():

    data = torch.load('embed.pt').numpy().T

    plt.figure(figsize=(8, 8))
    plt.scatter(data[0], data[1], 2)
    # plt.axis('off')
    plt.savefig('embed.png')


def cluster(dim):

    data = torch.load('embed_{}.pt'.format(dim)).numpy()

    plt.figure(figsize=(8, 8))

    data = TSNE(n_components=2).fit_transform(data)

    plt.scatter(data[:, 0], data[:, 1], 2)
    plt.savefig('embed_{}.png'.format(dim))


if __name__ == '__main__':

    main()

    # plot_points()

    cluster(8)

