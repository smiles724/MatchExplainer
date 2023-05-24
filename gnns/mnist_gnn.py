# from https://github.com/rusty1s/pytorch_geometric/blob/master/examples/mnist_voxel_grid.py
import os
import math
import random
import argparse
import os.path as osp

import torch
from functools import wraps
import torch.nn.functional as F
from torch_geometric.datasets import MNISTSuperpixels
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import SplineConv, voxel_grid, max_pool_x
from torch_geometric.data import Batch, Data
from torch_geometric.utils import subgraph

import sys

sys.path.append('..')
from utils import set_seed
from utils.logger import Logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train MNIST Spuer-Pixel Model")

    parser.add_argument('--data_path', nargs='?', default=osp.join(osp.dirname(__file__), '..', 'node_features', 'MNIST'), help='Input node_features path.')
    parser.add_argument('--model_path', nargs='?', default=osp.join(osp.dirname(__file__), '..', 'param', 'gnns'), help='path for saving trained model.')
    parser.add_argument('--cuda', type=int, default=0, help='GPU device.')
    parser.add_argument('--epoch', type=int, default=10, help='Number of epoch.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--no_select', action='store_true')
    parser.add_argument('--random_drop', action='store_true')
    parser.add_argument('--ratio', type=float, default=0.95)

    return parser.parse_args()


def overload(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        if len(args) + len(kargs) == 2:
            # for inputs like model(g)
            if len(args) == 2:
                g = args[1]
            # for inputs like model(g=g)
            else:
                g = kargs['node_features']
            return func(args[0], g)

        elif len(args) + len(kargs) == 6:
            # for inputs like model(x, ..., batch, pos)
            if len(args) == 6:
                _, x, edge_index, edge_attr, batch, pos = args
            # for inputs like model(x=x, ..., batch=batch, pos=pos)
            else:
                x, edge_index = kargs['x'], kargs['edge_index']
                edge_attr, batch = kargs['edge_attr'], kargs['batch']
                pos = kargs['pos']
            row, col = edge_index
            ground_tuth_mask = (x[row] > 0).view(-1).bool() * (x[col] > 0).view(-1).bool()
            g = Batch(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch, pos=pos, ground_tuth_mask=ground_tuth_mask).to(x.device)
            return func(args[0], g)
        else:
            raise TypeError

    return wrapper


class MNISTNet(torch.nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = SplineConv(1, 32, dim=2, kernel_size=5)
        self.conv2 = SplineConv(32, 64, dim=2, kernel_size=5)
        self.conv3 = SplineConv(64, 64, dim=2, kernel_size=5)
        self.fc1 = torch.nn.Linear(4 * 64, 128)
        self.fc2 = torch.nn.Linear(128, 10)
        import torch_geometric.transforms as T
        self.transform = T.Cartesian(cat=False, max_value=9)

    @overload
    def forward(self, data):
        graph_x, node_x = self.get_graph_rep(data)
        pred = self.get_pred(graph_x)
        self.readout = pred.softmax(dim=1)
        return pred, node_x

    @overload
    def get_graph_rep(self, data):
        x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        x = F.elu(self.conv2(x, data.edge_index, data.edge_attr))
        node_x = F.elu(self.conv3(x, data.edge_index, data.edge_attr))

        cluster = voxel_grid(data.pos, data.batch, size=14, start=0, end=27.99)
        x, _ = max_pool_x(cluster, node_x, data.batch, size=4)
        graph_x = x.view(-1, self.fc1.weight.size(1))
        return graph_x, node_x

    def get_pred(self, graph_x):
        graph_x = F.elu(self.fc1(graph_x))
        graph_x = F.dropout(graph_x, training=self.training)
        graph_x = self.fc2(graph_x)
        return F.log_softmax(graph_x, dim=1)


def train(epoch, loader):
    model.train()
    if epoch == 6:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001
    if epoch == 16:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        F.nll_loss(model(data)[0], data.y).backward()
        optimizer.step()


def test():
    model.eval()
    correct = 0

    for data in test_loader:
        data = data.to(device)
        pred = model(data)[0].max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(test_dataset)


def select(n_sample=1):
    candidate = {}
    for i in range(10): candidate[i] = []
    for g in finetune_loader: candidate[g.y.item()].append(g)

    data = []
    for g in finetune_loader:
        g.to(device)
        g_ = g.clone()
        model.eval()
        with torch.no_grad():
            output, node_x = model(g)  # what if we consider graphs whose predictions are not correct?

        K = math.ceil(args.ratio * len(node_x))  # batch_norm does not allow batch size=1 and requires K>1
        n = 0
        random.shuffle(candidate[g.y.item()])  # random sampling
        for g_counter in candidate[g.y.item()]:
            if len(g_counter.x) < K: continue  # skip counterpart graph with too few nodes
            g_counter.to(device)
            model.eval()
            with torch.no_grad():
                output, node_x_ = model(g_counter)

            if output.argmax(dim=1) != g_counter.y: continue  # choose the graph with the correct prediction

            node_x_ = node_x_.to(device)
            similarity = torch.cdist(node_x, node_x_)
            subg_idx = []
            while len(subg_idx) < K:
                min_idx = torch.argmin(similarity).item()
                row_idx = min_idx // len(node_x_)
                col_idx = min_idx % len(node_x_)

                similarity[row_idx] = 1e9
                similarity[:, col_idx] = 1e9
                subg_idx.append(row_idx)

            g = g_.clone()
            g.edge_index, g.edge_attr = subgraph(subg_idx, g.edge_index, g.edge_attr, num_nodes=g.num_nodes, relabel_nodes=True)  # we can directly set relabel=True
            pos = None
            try:
                pos = g.pos[subg_idx]
            except:
                pass
            g.pos = pos
            data.append(Data(x=g.x[subg_idx], pos=g.pos, edge_index=g.edge_index, edge_attr=g.edge_attr, y=g.y))
            n += 1
            if n >= n_sample: break
    return data


def drop(ratio=0.95):
    data = []
    for g in finetune_loader:
        nodes = [i for i in range(g.num_nodes)]
        subg_idx = random.sample(nodes, math.ceil(ratio * g.num_nodes))

        g.edge_index, g.edge_attr = subgraph(subg_idx, g.edge_index, g.edge_attr, num_nodes=g.num_nodes, relabel_nodes=True)
        pos = None
        try:
            pos = g.pos[subg_idx]
        except:
            pass
        g.pos = pos
        data.append(Data(x=g.x[subg_idx], pos=g.pos, edge_index=g.edge_index, edge_attr=g.edge_attr, y=g.y))

    return data


if __name__ == '__main__':
    set_seed(0)
    args = parse_args()
    logger = Logger.init_logger(filename="../log/mnist.log")
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    transform = T.Cartesian(cat=False, max_value=9)
    train_dataset = MNISTSuperpixels(args.data_path, True, transform=transform)
    test_dataset = MNISTSuperpixels(args.data_path, False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    finetune_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    d = train_dataset
    model = MNISTNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0
    for epoch in range(1, args.epoch):
        if epoch >= args.epoch * 0.1 and not args.no_select:
            if args.random_drop:
                sub_graphs = drop()
            else:
                sub_graphs = select()
            sub_loader = DataLoader(sub_graphs, batch_size=args.batch_size, shuffle=True)
            train(epoch, sub_loader)
        else:
            train(epoch, train_loader)
        test_acc = test()
        best_acc = max(best_acc, test_acc)
        logger.info('Epoch: {:02d}, Test: {:.4f}'.format(epoch, test_acc))

    if not osp.exists(args.model_path):
        os.makedirs(args.model_path)
    if args.no_select: torch.save(model.cpu(), osp.join(args.model_path, 'mnist_net.pt'))
    logger.info(f'Best acc: {best_acc}')
