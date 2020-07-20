import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

path = 'data/cit-HepPh/cit-hepPh/raw'

np.random.seed(147)
torch.manual_seed(147)


def gen_dataset():

    df = pd.read_csv(os.path.join(path, 'cit-clean.csv'), dtype=str, header=0)
    df = df.sample(frac=1, random_state=147)
    df = df.values

    cit = df[:, 1:].astype('float64')
    ids = df[:, 0]

    for i in range(cit.shape[0]):
        id = ids[i]
        year = '19'+id[:2] if id[0] == '9' else '20'+id[:2]
        pads = int(year) - 1992
        for j in range(pads):
            cit[i, j] = -1

    train_input = torch.from_numpy(cit[:20000, :-1])
    train_target = torch.from_numpy(cit[:20000, 1:])
    test_input = torch.from_numpy(cit[20000:, :-1])
    test_target = torch.from_numpy(cit[20000:, 1:])

    return train_input, train_target, test_input, test_target, ids


class SeqNet(nn.Module):
    def __init__(self):
        super(SeqNet, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear =nn.Linear(51, 1)

    def forward(self, x, future=0):
        outputs = []
        h_t = torch.zeros(x.size(0), 51, dtype=torch.double)
        c_t = torch.zeros(x.size(0), 51, dtype=torch.double)
        h_t2 = torch.zeros(x.size(0), 51, dtype=torch.double)
        c_t2 = torch.zeros(x.size(0), 51, dtype=torch.double)

        for i, input_t in enumerate(x.chunk(x.size(1), dim=1)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):  # if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


def main():

    epoch = 10
    train_input, train_target, test_input, test_target, _ = gen_dataset()

    net = SeqNet()
    net.double()
    # net.load_state_dict(torch.load('pretrained/lstm.pt'))
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(net.parameters(), lr=1)
    # optimizer = optim.LBFGS(net.parameters(), lr=1)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * epoch, 0.75 * epoch],
    #                                                  gamma=0.3)

    for i in range(epoch):

        print('Epoch: ', i)

        def closure():
            optimizer.zero_grad()
            out = net(train_input)
            loss = criterion(out, train_target)
            print('train loss:', loss.item())
            loss.backward()
            return loss
        optimizer.step(closure)

        # begin to predict, no need to track gradient here
        with torch.no_grad():
            future = 10
            pred = net(test_input, future=future)
            loss = criterion(pred[:, :-future], test_target)
            print('test loss:', loss.item())
            y = pred.detach().numpy()

        # scheduler.step()
    torch.save(net.state_dict(), 'pretrained/lstm_lbfgs.pt')


if __name__ == '__main__':
    main()
