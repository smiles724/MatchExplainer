import time
import math
import numpy as np

import os
import os.path as osp
from tqdm import tqdm

import torch
from torch_geometric.data import DataLoader
from torch_geometric.utils import subgraph

from gnns import *
from datasets.graphss2_dataset import get_dataloader
from utils import set_seed
from utils.parser import parse_args
from utils.dataset import get_datasets
from utils.logger import Logger


np.set_printoptions(precision=2, suppress=True)
folder_dict = {'mutag': 'MUTAG', 'ba3': 'BA3', 'mnist': 'MNIST', 'vg': 'VG'}
n_classes_dict = {'mutag': 2, 'mnist': 10, 'ba3': 3, 'vg': 5}

args = parse_args()
set_seed(args.random_seed)
folder = osp.join(args.data_root, folder_dict[args.dataset])
if args.recall: args.dataset = 'ba3'

# load GNNs to be explained
path = osp.join(args.param_root, 'gnns/%s_net.pt' % args.dataset)
train_dataset, val_dataset, test_dataset = get_datasets(args.dataset, root=args.data_root)
device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
os.makedirs(args.log_root, exist_ok=True)
logger = Logger.init_logger(filename=args.log_root + f"/match-{args.dataset}.log")

if args.dataset == 'graphsst2':
    dataloader = get_dataloader(train_dataset, batch_size=args.batch_size, random_split_flag=True,
                                data_split_ratio=[0.8, 0.1, 0.1], seed=2)
    train_loader, val_loader, test_loader = dataloader['train'], dataloader['eval'], dataloader['test']
else:
    # filter graphs with right prediction
    for label, dataset in zip(['train', 'val', 'test'], [train_dataset, val_dataset, test_dataset]):
        batch_size = 1 if label == 'test' else args.batch_size
        dataset_mask = []
        flitered_path = osp.join(args.param_root, f"filtered/{args.dataset}_idx_{label}.pt")
        if osp.exists(flitered_path):
            graph_mask = torch.load(flitered_path)
        else:
            os.makedirs(osp.join(args.param_root, "filtered"), exist_ok=True)
            loader = DataLoader(dataset, batch_size=1, shuffle=False)
            model = torch.load(path).to(device)
            graph_mask = torch.zeros(len(loader.dataset), dtype=torch.bool)
            idx = 0
            for g in tqdm(iter(loader), total=len(loader)):
                g.to(device)
                model(g)
                if g.y == model.readout.argmax(dim=1): graph_mask[idx] = True
                idx += 1
            torch.save(graph_mask, flitered_path)
            dataset_mask.append(graph_mask)

        logger.info("number of graphs(%s): %4d" % (label, graph_mask.nonzero(as_tuple=False).size(0)))
        exec("%s_loader = DataLoader(dataset[graph_mask], batch_size=%d, shuffle=False, drop_last=False)" % (
            label, batch_size))

model = torch.load(path).to(device)
model.eval()
t0 = time.time()

# generate the reference set
candidate = {}
for i in range(n_classes_dict[args.dataset]): candidate[i] = []
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

# examine the performance of MatchExplainer
test_acc = []
top_ratio_list = [0.1 * i for i in range(1, 11)]
with torch.no_grad():
    for top_ratio in top_ratio_list:
        acc, recall = [], []

        if top_ratio == 1.0:
            test_acc.append(1.0)  # the accuracy of the entire graph is 1.0
            continue

        for g in test_loader:
            g.to(device)
            g_ = g.clone()
            _, node_x = model(g)
            node_idx = []

            K = math.ceil(top_ratio * len(node_x))
            if args.recall: K = args.recall_k
            flag = 0
            recall_tmp = []
            loop = candidate[g.y.item()]
            if args.dataset == 'mnist':
                loop = loop[:int(0.1 * len(loop))]    # select part of available set as the counterpart
            # elif args.dataset == 'ba3':
            #     loop = loop[:int(0.7 * len(loop))]

            for node_x_ in loop:
                if len(node_x_) < K: continue  # give up counterpart graph with too few nodes
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

                if args.recall:
                    if isinstance(g_.ground_truth_mask, list):
                        g_.ground_truth_mask = g_.ground_truth_mask[0]
                    gt_edge = g_.edge_index[:, torch.tensor(g_.ground_truth_mask).bool()]
                    gt_node = torch.unique(gt_edge).tolist()
                    recall_tmp.append(sum([1 if x in gt_node else 0 for x in subg_idx]) / len(gt_node))
                    continue
                g = g_.clone()
                g.edge_index, g.edge_attr = subgraph(subg_idx, g.edge_index, g.edge_attr,
                                                     num_nodes=g.num_nodes, relabel_nodes=True)
                pos = None
                try:
                    pos = g.pos[subg_idx]
                except:
                    pass
                g.pos = pos

                g.x = g.x[subg_idx]
                g.batch = g.batch[subg_idx]
                out, _ = model(g)

                res_acc = (g.y == model.readout.argmax(dim=1))
                if res_acc:
                    acc.append(1)
                    flag = 1
                    break

            if flag == 0: acc.append(0)
            if args.recall: recall.append(max(recall_tmp))
        if not args.recall:
            test_acc.append(sum(acc) / len(acc))
            logger.info(f'Accuracy for {top_ratio:.1f}: {test_acc[-1]}.')
        else:
            logger.info(f'Recall for top-{args.recall_k}: {sum(recall) / len(recall)}.'); break
logger.info(f'Total time: {time.time() - t0}')
if not args.recall:
    logger.info(f'Testing Finished. \nACC: {[round(x, 2) for x in test_acc]}\nAUC:{round(sum(test_acc) / len(test_acc), 4)}')


