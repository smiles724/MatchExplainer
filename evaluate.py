import os
import json
import argparse

import torch
import numpy as np
from tqdm import tqdm
from utils.dataset import get_datasets
from explainers import *
from torch_geometric.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain MatchExplainer")
    parser.add_argument('--cuda', type=int, default=0, help='GPU device.')
    parser.add_argument('--dataset', type=str, default='ba3', choices=['mutag', 'ba3', 'graphsst2', 'mnist', 'vg', 'reddit5k'])
    parser.add_argument('--result_dir', type=str, default="results/", help='Result directory.')
    parser.add_argument('--lr', type=float, default=2 * 1e-4, help='Fine-tuning learning rate.')
    parser.add_argument('--epoch', type=int, default=20, help='Fine-tuning rpoch.')
    return parser.parse_args()


args = parse_args()
results = {}
if args.dataset == 'ba3':
    ground_truth = True
else:
    ground_truth = False
device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
train_dataset, val_dataset, test_dataset = get_datasets(name=args.dataset)
graph_mask = torch.load(f'param/filtered/{args.dataset}_idx_test.pt')
test_loader = DataLoader(test_dataset[graph_mask], batch_size=1, shuffle=False, drop_last=False)
ratios = [0.1 * i for i in range(1, 11)]

refine = torch.load(f'param/refine/{args.dataset}.pt')
refine.remap_device(device)

# -------------------------------------------------------
print("Evaluate MatchExplainer-FT w.r.t. ACC-AUC (& Recall@5)...")
acc_logger, recall_logger = [], []
for g in tqdm(iter(test_loader), total=len(test_loader)):
    g = g.to(device)
    refine.explain_graph(g, fine_tune=False)
    acc_logger.append(refine.evaluate_acc(ratios)[0])
    if ground_truth:
        recall_logger.append(refine.evaluate_recall(topk=5))

results["MatchExplainer-FT"] = {"ROC-AUC": list(np.array(acc_logger).mean(axis=0)[0]), "ACC-AUC": np.array(acc_logger).mean(axis=0).mean(),
                                "Recall@5": "nan" if not ground_truth else np.array(recall_logger).mean(axis=0)}

# ---------------------------------------------------
print("Evaluate MatchExplainer w.r.t. ACC-AUC...")
recall_logger, tuned = [], []
results["MatchExplainer"] = {}
for i, r in enumerate(ratios):
    acc_logger = []
    for g in tqdm(iter(test_loader), total=len(test_loader)):
        g.to(device)
        refine.explain_graph(g, fine_tune=True, ratio=r, lr=args.lr, epoch=args.epoch)
        acc_logger.append(refine.evaluate_acc(ratios)[0])
    results["MatchExplainer"]["R-%.2f" % r] = {"ROC-AUC": list(np.array(acc_logger).mean(axis=0)[0]), "ACC-AUC": np.array(acc_logger).mean(axis=0).mean(), }
    tuned.append(np.array(acc_logger).mean(axis=0)[0, i])
results["MatchExplainer"]["ROC-AUC"] = list(tuned)
results["MatchExplainer"]["ACC-AUC"] = np.mean(tuned)

# ---------------------------------------------------
if ground_truth:
    print("Evaluate MatchExplainer w.r.t. Recall@5...")
    for g in tqdm(iter(test_loader), total=len(test_loader)):
        g.to(device)
        refine.explain_graph(g, fine_tune=True, ratio=0.3, lr=1e-4, epoch=20)
        recall_logger.append(refine.evaluate_recall(topk=5))
    results["MatchExplainer"]["Recall@5"] = np.mean(recall_logger)

print(results)
os.makedirs(args.result_dir, exist_ok=True)
with open(os.path.join(args.result_dir, f"{args.dataset}_results.json"), "w") as f:
    json.dump(results, f, indent=4)
