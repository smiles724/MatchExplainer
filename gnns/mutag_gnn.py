import sys
import time
import random
import argparse
import os
import os.path as osp

import torch
import torch.nn as nn
from torch.nn import ModuleList
from torch.nn import Sequential as Seq, ReLU, Linear as Lin, Softmax
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINEConv, BatchNorm, global_mean_pool, GCNConv

sys.path.append('..')
from gnns.overloader import overload
from datasets.mutag_dataset import Mutagenicity
from utils import set_seed, Gtrain, Gtest, Gselect
from utils.train_utils import Gdrop
from utils.logger import Logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train Mutag Model")
    parser.add_argument('--data_path', nargs='?',
                        default=osp.join(osp.dirname(__file__), '..', 'node_features', 'MUTAG'),
                        help='Input node_features path.')
    parser.add_argument('--model_path', nargs='?', default=osp.join(osp.dirname(__file__), '..', 'param', 'gnns'),
                        help='path for saving trained model.')
    parser.add_argument('--cuda', type=int, default=0, help='GPU device.')
    parser.add_argument('--epoch', type=int, default=200, help='Number of epoch.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--verbose', type=int, default=10, help='Interval of evaluation.')
    parser.add_argument('--num_unit', type=int, default=2, help='number of Convolution layers(units)')  # dropnode适用于更深的layer
    parser.add_argument('--no_select', action='store_true')
    parser.add_argument('--random_drop', action='store_true')
    parser.add_argument('--ratio', type=float, default=0.95)
    return parser.parse_args()


class MutagNet(torch.nn.Module):
    def __init__(self, conv_unit=2):
        super(MutagNet, self).__init__()
        self.node_emb = Lin(14, 32)
        self.edge_emb = Lin(3, 32)
        self.relu_nn = ModuleList([ReLU() for _ in range(conv_unit)])
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.relus = ModuleList()

        for i in range(conv_unit):
            conv = GINEConv(nn=Seq(Lin(32, 75), self.relu_nn[i], Lin(75, 32)))  # GIN
            # conv = GCNConv(32, 32)     # GCN performs similarly to GIN
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(32))
            self.relus.append(ReLU())

        self.lin1 = Lin(32, 16)
        self.relu = ReLU()
        self.lin2 = Lin(16, 2)
        self.softmax = Softmax(dim=1)

    @overload
    def forward(self, x, edge_index, edge_attr, batch):
        node_x = self.get_node_reps(x, edge_index, edge_attr, batch)
        graph_x = global_mean_pool(node_x, batch)
        return self.get_pred(graph_x), node_x

    @overload
    def get_node_reps(self, x, edge_index, edge_attr, batch):
        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)
        for conv, batch_norm, ReLU in zip(self.convs, self.batch_norms, self.relus):
            x = conv(x, edge_index, edge_attr)   # GIN
            # x = conv(x, edge_index)   # GCN
            x = ReLU(batch_norm(x))
        node_x = x
        return node_x

    @overload
    def get_graph_rep(self, x, edge_index, edge_attr, batch):
        node_x = self.get_node_reps(x, edge_index, edge_attr, batch)
        graph_x = global_mean_pool(node_x, batch)
        return graph_x

    def get_pred(self, graph_x):
        pred = self.relu(self.lin1(graph_x))
        pred = self.lin2(pred)
        self.readout = self.softmax(pred)
        return pred

    def reset_parameters(self):
        with torch.no_grad():
            for param in self.parameters():
                param.uniform_(-1.0, 1.0)


if __name__ == '__main__':
    set_seed(0)
    args = parse_args()
    logger = Logger.init_logger(filename="../log/mutag.log")
    if not args.no_select: logger.info(f'Ratio: {args.ratio}')

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    test_dataset = Mutagenicity(args.data_path, mode='testing')
    val_dataset = Mutagenicity(args.data_path, mode='evaluation')
    train_dataset = Mutagenicity(args.data_path, mode='training')

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    select_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    model = MutagNet(args.num_unit).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=20, min_lr=1e-4)
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
                sub_graphs = Gselect('mutag', select_loader, model, device=device, ratio=args.ratio)
                sub_loader = DataLoader(sub_graphs, batch_size=args.batch_size, shuffle=True)
            loss = Gtrain(sub_loader, model, optimizer, device=device, criterion=nn.CrossEntropyLoss())
            _, train_acc = Gtest(sub_loader, model, device=device, criterion=nn.CrossEntropyLoss())
        else:
            # at first do not fine-tune, because GNN is not robust
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
    if args.no_select: torch.save(model.cpu(), osp.join(args.model_path, 'mutag_net.pt'))
    logger.info(f'Best test acc: {best_acc:.5f}')
