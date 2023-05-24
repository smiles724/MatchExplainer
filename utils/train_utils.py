import math
import random
import torch
from torch import nn
from torch_geometric.utils import subgraph
from torch_geometric.data import Data


# General function for training g classification(regresion) task and node classification task under multiple graphs.
def Gtrain(train_loader, model, optimizer, device, criterion=nn.MSELoss()):
    model.train()
    loss_all = 0
    criterion = criterion

    for data in train_loader:
        data.to(device)
        optimizer.zero_grad()
        out, _ = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()

    return loss_all / len(train_loader.dataset)


def Gtest(test_loader, model, device, criterion=nn.L1Loss(reduction='mean'), ):
    model.eval()
    error = 0
    correct = 0

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output, _ = model(data.x, data.edge_index, data.edge_attr, data.batch, )

            error += criterion(output, data.y) * data.num_graphs
            correct += float(output.argmax(dim=1).eq(data.y).sum().item())

        return error / len(test_loader.dataset), correct / len(test_loader.dataset)


def Gselect(dataset, finetune_loader, model, device, ratio=0.1, n_sample=1):
    n_classes_dict = {'mutag': 2, 'mnist': 10, 'ba3': 3, 'vg': 5}
    candidate = {}
    for i in range(n_classes_dict[dataset]): candidate[i] = []
    for g in finetune_loader: candidate[g.y.item()].append(g)

    data = []
    for g in finetune_loader:
        g.to(device)
        g_ = g.clone()
        model.eval()  # eval模式对正确的预测很关键
        with torch.no_grad():
            output, node_x = model(g)  # 不必只考虑预测正确的样本？

        K = math.ceil(ratio * len(node_x))  # batch_norm不适用于单个点，另外单个点的输入也有问题（K>1）
        n = 0
        random.shuffle(candidate[g.y.item()])  # 随机采样
        for g_counter in candidate[g.y.item()]:
            if len(g_counter.x) < K: continue  # 不考虑节点数过小的配对
            g_counter.to(device)
            model.eval()
            with torch.no_grad():
                output, node_x_ = model(g_counter)

            if output.argmax(dim=1) != g_counter.y: continue  # 选预测正确的样本作为配对

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
            g.edge_index, g.edge_attr = subgraph(subg_idx, g.edge_index, g.edge_attr,
                                                 num_nodes=g.num_nodes, relabel_nodes=True)  # 可以直接设置relabel=True
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


def Gdrop(dataloader, ratio=0.9):
    data = []
    for g in dataloader:
        nodes = [i for i in range(g.num_nodes)]
        subg_idx = random.sample(nodes, math.ceil(ratio * g.num_nodes))

        g.edge_index, g.edge_attr = subgraph(subg_idx, g.edge_index, g.edge_attr,
                                             num_nodes=g.num_nodes, relabel_nodes=True)
        pos = None
        try:
            pos = g.pos[subg_idx]
        except:
            pass
        g.pos = pos

        data.append(Data(x=g.x[subg_idx], pos=g.pos, edge_index=g.edge_index, edge_attr=g.edge_attr, y=g.y))

    return data
