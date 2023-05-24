import time
import copy
import random
import argparse

import os
import os.path as osp
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import Linear as Lin, Softmax
from torch_geometric.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import APPNP, BatchNorm, global_mean_pool

import sys

sys.path.append('..')
from gnns.overloader import overload
from datasets.vg_dataset import Visual_Genome
from utils import set_seed, Gtrain, Gtest, Gselect
from utils.train_utils import Gdrop
from utils.logger import Logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train Visual Genome Model")
    parser.add_argument('--data_path', nargs='?', default=osp.join(osp.dirname(__file__), '..', 'data', 'VG'),
                        help='Input node_features path.')
    parser.add_argument('--model_path', nargs='?', default=osp.join(osp.dirname(__file__), '..', 'param', 'gnns'),
                        help='path for saving trained model.')
    parser.add_argument('--cuda', type=int, default=0, help='GPU device.')
    parser.add_argument('--epoch', type=int, default=100, help='Number of epoch.')
    parser.add_argument('--lr', type=float, default=0.5 * 1e-3, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--verbose', type=int, default=10, help='Interval of evaluation.')
    parser.add_argument('--no_select', action='store_true')
    parser.add_argument('--random_drop', action='store_true')
    parser.add_argument('--ratio', type=float, default=0.95)
    return parser.parse_args()


class VGNet(torch.nn.Module):

    def __init__(self):
        super(VGNet, self).__init__()
        self.node_encode = nn.Sequential(OrderedDict([  # With square kernels and equal stride
            ('conv1', nn.Conv2d(3, 8, 3, stride=1)), ('relu1', nn.ReLU()), ('conv2', nn.Conv2d(8, 5, 5, stride=2)),
            ('relu2', nn.ReLU()), ('flatten', nn.Flatten()), ('lin1', Lin(245, 128)), ('relu3', nn.ReLU()), ]))

        # set add_self_loops to False for torch > 1.5
        self.conv3 = APPNP(K=2, alpha=0.8, add_self_loops=False)
        self.norm = BatchNorm(128)
        self.mlp = nn.Sequential(OrderedDict([('lin2', Lin(128, 64)), ('relu3', nn.ReLU()), ('lin3', Lin(64, 5))]))

        self.softmax = Softmax(dim=1)

    @overload
    def forward(self, x, edge_index, edge_attr, batch):
        graph_x, node_x = self.get_graph_rep(x, edge_index, edge_attr, batch)
        return self.get_pred(graph_x), node_x

    @overload
    def get_node_reps(self, x, edge_index, edge_attr, batch):
        edge_weight = edge_attr.view(-1)
        x = self.node_encode(x)  # x: (3, 20, 20)
        x = self.conv3(x, edge_index, edge_weight)
        x = self.norm(x)
        return x

    @overload
    def get_graph_rep(self, x, edge_index, edge_attr, batch):
        node_x = self.get_node_reps(x, edge_index, edge_attr, batch)
        graph_x = global_mean_pool(node_x, batch)
        return graph_x, node_x

    def get_pred(self, graph_x):
        pred = self.mlp(graph_x)
        self.readout = self.softmax(pred)
        return pred

    def reset_parameters(self):
        self.reserve = []
        with torch.no_grad():
            for param in self.parameters():
                self.reserve.append(copy.deepcopy(param))
                param.uniform_(-1.0, 1.0)

    def restore_parameters(self):
        with torch.no_grad():
            for i, param in enumerate(self.parameters()):
                param.set_(self.reserve[i])


if __name__ == '__main__':
    set_seed(0)
    args = parse_args()
    logger = Logger.init_logger(filename="../log/vg.log")
    if not args.no_select: logger.info(f'Ratio: {args.ratio}')

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    test_dataset = Visual_Genome(args.data_path, mode='testing')
    val_dataset = Visual_Genome(args.data_path, mode='evaluation')
    train_dataset = Visual_Genome(args.data_path, mode='training')

    model = VGNet().to(device)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True)
    select_loader = DataLoader(train_dataset, batch_size=1, pin_memory=True, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, min_lr=1e-5)

    min_error = None
    best_acc = 0
    for epoch in range(1, args.epoch + 1):
        t1 = time.time()
        lr = scheduler.optimizer.param_groups[0]['lr']

        if epoch >= args.epoch * 0.1 and not args.no_select:
            if args.random_drop:
                sub_graphs = Gdrop(select_loader)
                sub_loader = DataLoader(sub_graphs, batch_size=args.batch_size, shuffle=True)
            else:
                sub_graphs = Gselect('vg', select_loader, model, device=device, ratio=args.ratio)
                sub_loader = DataLoader(sub_graphs, batch_size=args.batch_size, shuffle=True)
            loss = Gtrain(sub_loader, model, optimizer, device=device, criterion=nn.CrossEntropyLoss())
            _, train_acc = Gtest(sub_loader, model, device=device, criterion=nn.CrossEntropyLoss())
        else:
            loss = Gtrain(train_loader, model, optimizer, device=device, criterion=nn.CrossEntropyLoss())
            _, train_acc = Gtest(train_loader, model, device=device, criterion=nn.CrossEntropyLoss())

        val_error, val_acc = Gtest(val_loader, model, device=device, criterion=nn.CrossEntropyLoss())
        test_error, test_acc = Gtest(test_loader, model, device=device, criterion=nn.CrossEntropyLoss())
        scheduler.step(val_error)
        if min_error is None or val_error <= min_error:
            min_error = val_error

        t2 = time.time()
        if epoch % args.verbose == 0:
            test_error, test_acc = Gtest(test_loader, model, device=device, criterion=nn.CrossEntropyLoss())
            best_acc = max(best_acc, test_acc)
            t3 = time.time()
            logger.info('Epoch{:4d}[{:.3f}s]: LR: {:.5f}, Loss: {:.5f}, Test Loss: {:.5f}, '
                        'Test acc: {:.5f}'.format(epoch, t3 - t1, lr, loss, test_error, test_acc))
            continue

        logger.info('Epoch{:4d}[{:.3f}s]: LR: {:.5f}, Loss: {:.5f}, Train acc: {:.5f}, Validation Loss: {:.5f}, '
                    'Validation acc: {:5f}'.format(epoch, t2 - t1, lr, loss, train_acc, val_error, val_acc))

    if not osp.exists(args.model_path):
        os.makedirs(args.model_path)
    torch.save(model.cpu(), osp.join(args.model_path, 'vg_net.pt'))
    logger.info(f'Best test acc: {best_acc:.5f}')

