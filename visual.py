import argparse
import os.path as osp
import io
import math
from PIL import Image
from rdkit.Chem.Draw import rdMolDraw2D

import torch
from torch_geometric.utils import subgraph
from torch_geometric.data import DataLoader
import sys

sys.path.append('..')
from gnns import *
from datasets.mutag_dataset import Mutagenicity
from explainers.visual import graph_to_mol


def parse_args():
    parser = argparse.ArgumentParser(description="Train Mutag Model")
    parser.add_argument('--data_path', nargs='?', default=osp.join(osp.dirname(__file__), 'node_features', 'MUTAG'),
                        help='Input node_features path.')
    parser.add_argument('--model_path', nargs='?', default=osp.join(osp.dirname(__file__), 'param', 'gnns'),
                        help='path for saving trained model.')
    parser.add_argument('--param_root', type=str, default="param/")
    parser.add_argument('--dataset', type=str, default='mutag',
                        choices=['mutag', 'ba3', 'graphsst2', 'mnist', 'vg', 'reddit5k'])

    parser.add_argument('--n_sample', type=int, default=300)
    parser.add_argument('--cuda', type=int, default=0, help='GPU device.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--verbose', type=int, default=100, help='Interval of evaluation.')
    parser.add_argument('--ratio', type=float, default=0.5)
    return parser.parse_args()


args = parse_args()
test_dataset = Mutagenicity(args.data_path, mode='testing')
val_dataset = Mutagenicity(args.data_path, mode='evaluation')
train_dataset = Mutagenicity(args.data_path, mode='training')

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset[:args.n_sample], batch_size=1, shuffle=False)
device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

path = osp.join(args.param_root, 'gnns/%s_net.pt' % args.dataset)
model = torch.load(path).to(device)
model.eval()

candidate = {}
for i in range(2): candidate[i] = []  # 2 classes for mutag
with torch.no_grad():
    for g in train_loader:
        g.to(device)
        _, node_x = model(g)

        for i in torch.unique(g.batch):
            node_indicator = torch.where(g.batch == i)[0].detach().cpu()
            candidate[g.y[i].item()].append(node_x[node_indicator].cpu())

    for g in val_loader:
        g.to(device)
        _, node_x = model(g)

        for i in torch.unique(g.batch):
            node_indicator = torch.where(g.batch == i)[0].detach().cpu()
            candidate[g.y[i].item()].append(node_x[node_indicator].cpu())

for i, g in enumerate(test_loader):
    flag = 0
    g.to(device)
    g_ = g.clone()
    _, node_x = model(g)
    node_idx = []

    K = math.ceil(args.ratio * len(node_x))
    loop = candidate[g.y.item()]
    for node_x_ in loop:
        if len(node_x_) < K: continue
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
        g.edge_index, g.edge_attr = subgraph(subg_idx, g.edge_index, g.edge_attr, num_nodes=g.num_nodes,
                                             relabel_nodes=True)
        pos = None
        try:
            pos = g.pos[subg_idx]
        except:
            pass
        g.pos = pos
        g.x = g.x[subg_idx]
        g.batch = g.batch[subg_idx]
        model(g)

        conf = model.readout[:, g.y]
        if conf > 0.99:
            flag = 1
            break

    if flag == 0: continue
    x = g_.x.detach().cpu().tolist()
    mol = graph_to_mol(x, g_.edge_index.T.detach().cpu().tolist(), g_.edge_attr.detach().cpu().tolist())
    d = rdMolDraw2D.MolDraw2DCairo(500, 500)
    edge_index, _ = subgraph(subg_idx, g_.edge_index, num_nodes=g_.num_nodes)
    hit_bonds = []
    for (u, v) in edge_index.T:
        hit_bonds.append(mol.GetBondBetweenAtoms(int(u), int(v)).GetIdx())
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=subg_idx, highlightBonds=hit_bonds,
                                       highlightAtomColors={i: (0, 1, 0) for i in subg_idx},
                                       highlightBondColors={i: (0, 1, 0) for i in hit_bonds})
    d.FinishDrawing()
    d.WriteDrawingText(f'images/image_y{g.y.item()}_{round(conf.item(), 5)}.png')
