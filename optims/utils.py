
import torch

import torch_geometric.transforms as T
from torch_geometric.data.data import Data
from torch_geometric.nn import LGConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from collections import defaultdict

from freerec.data.datasets.base import RecDataSet
from freerec.data.tags import USER, ITEM, ID


class Smoother:

    def __init__(
        self, graph: Data, 
        beta: float, L: int, aggr: str
    ) -> None:
        self.graph = graph
        self.beta = beta
        self.L = L
        self.aggr = aggr
        self.conv = LGConv(normalize=False)

    def aggregator(self, features: torch.Tensor):
        return self.conv(features, self.graph.adj_t)

    @torch.no_grad()
    def __call__(self, features: torch.Tensor):
        smoothed = features
        if self.aggr == 'neumann':
            norm_correction = 1 - self.beta ** (self.L + 1)     
            for _ in range(self.L):
                features = self.aggregator(features) * self.beta
                smoothed = smoothed + features
            smoothed = smoothed.mul(1 - self.beta).div(norm_correction)
        elif self.aggr == 'momentum':
            for _ in range(self.L):
                smoothed = self.aggregator(smoothed) * self.beta + features * (1 - self.beta)
        elif self.aggr == 'average':
            smoothed = features / (self.L + 1)
            for _ in range(self.L):
                features = self.aggregator(features)
                smoothed += features / (self.L + 1)
        else:
            raise ValueError(f"aggr should be average|neumann|momentum but {self.aggr} received ...")
        return smoothed


def get_item_graph(
    dataset: RecDataSet, 
    NUM_PADS: int = 0,
    receptive_field: int = 1,
    maxlen: int = None
):
    Item = dataset.fields[ITEM, ID]
    seqs = dataset.train().to_seqs(keepid=False)
    edge = defaultdict(int)
    neighbors = receptive_field
    for seq in seqs:
        seq = seq[-maxlen:]
        for i in range(len(seq) - 1):
            x = seq[i] + NUM_PADS
            for k, j in enumerate(range(i + 1, min(i + neighbors + 1, len(seq))), start=1):
                y = seq[j] + NUM_PADS
                edge[(x, y)] += 1. / k
                edge[(y, x)] += 1. / k

    edge_index, edge_weight = zip(*edge.items())
    edge_index = torch.LongTensor(
        edge_index
    ).t()
    graph = Data()
    graph.x = torch.empty((Item.count + NUM_PADS, 0))
    graph.edge_index = edge_index
    graph.edge_weight = torch.tensor(edge_weight, dtype=torch.float)
    T.ToSparseTensor(attr='edge_weight', remove_edge_index=True)(graph)
    graph.adj_t = gcn_norm(graph.adj_t, num_nodes=Item.count + NUM_PADS, add_self_loops=False)
    return graph

def get_user_item_graph(
    dataset: RecDataSet, NUM_PADS: int = 0
):
    from torch_geometric.utils import to_undirected
    User = dataset.fields[USER, ID]
    Item = dataset.fields[ITEM, ID]
    graph = dataset.train().to_heterograph(((USER, ID), '2', (ITEM, ID)))
    graph[User.name].x = torch.empty((User.count, 0), dtype=torch.long)
    graph[Item.name].x = torch.empty((Item.count + NUM_PADS, 0), dtype=torch.long)
    graph[User.name, '2', Item.name].edge_index[1].add_(NUM_PADS)
    graph = graph.to_homogeneous()
    graph.edge_index = to_undirected(graph.edge_index)
    T.ToSparseTensor()(graph)
    graph.adj_t = gcn_norm(
        graph.adj_t, num_nodes=User.count + Item.count + NUM_PADS,
        add_self_loops=False #
    )
    return graph

def get_graph(
    cfg,
    dataset: RecDataSet, 
    NUM_PADS: int = None,
    itemonly: bool = True
):
    if itemonly:
        graph = get_item_graph(dataset, NUM_PADS, receptive_field=cfg.H, maxlen=cfg.maxlen4graph)
    else:
        graph = get_user_item_graph(dataset, NUM_PADS)
    return graph