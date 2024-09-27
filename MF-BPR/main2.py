

import torch
import torch.nn as nn

import freerec
from freerec.data.fields import FieldModuleList
from freerec.data.tags import USER, SESSION, ITEM, TIMESTAMP, ID

import sys
sys.path.append("..")
from optims.models import UserItemArch, ItemOnlyArch, get_optimizer
from optims.utils import get_graph, Smoother

freerec.declare(version="0.4.3")

cfg = freerec.parser.Parser()
cfg.add_argument("-eb", "--embedding-dim", type=int, default=64)
cfg.add_argument("--maxlen4graph", type=int, default=50)

cfg.add_argument("--aggr", type=str, choices=('neumann', 'iterative'), default='neumann')
cfg.add_argument("--L", type=int, default=3, help="the number of layers for approximation")
cfg.add_argument("--beta3", type=float, default=0.9, help="the beta hyper-parameter")
cfg.add_argument("--H", type=int, default=1, help="the maximum walk length allowing for a pair of neighbors")

cfg.set_defaults(
    description="MF-BPR",
    root="../../data",
    dataset='MovieLens1M_550_Chron',
    epochs=500,
    batch_size=2048,
    optimizer='adam',
    lr=1e-3,
    weight_decay=1e-8,
    seed=1
)
cfg.compile()


class BPRMF(ItemOnlyArch):

    def __init__(self, fields: FieldModuleList) -> None:
        super().__init__()

        self.fields = fields
        self.User, self.Item = self.fields[USER, ID], self.fields[ITEM, ID]

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=1.e-4)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def marked_params(self, cfg, graph):
        other = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'fields' in name:
                continue
            else:
                other.append(param)
        User = self.fields[USER, ID]
        Item = self.fields[ITEM, ID]
        params = [
            {'params': Item.parameters(), 'smoother': Smoother(graph, beta=cfg.beta3, L=cfg.L, aggr=cfg.aggr)},
            {'params': User.parameters()},
        ]
        return params

    def predict(self, users: torch.Tensor, items: torch.Tensor):
        userEmbs = self.User.look_up(users) # B x 1 x D
        itemEmbs = self.Item.look_up(items) # B x n x D
        return torch.mul(userEmbs, itemEmbs).sum(-1)

    def recommend_from_full(self):
        return self.User.embeddings.weight, self.Item.embeddings.weight


class CoachForBPRMF(freerec.launcher.GenCoach):

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            users, positives, negatives = [col.to(self.device) for col in data]
            items = torch.cat(
                [positives, negatives], dim=1
            )
            scores = self.model.predict(users, items)
            pos, neg = scores[:, 0], scores[:, 1]
            loss = self.criterion(pos, neg)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss.item(), n=scores.size(0), mode="mean", prefix='train', pool=['LOSS'])


def main():

    dataset = getattr(freerec.data.datasets.sequential, cfg.dataset)(cfg.root)
    User, Item = dataset.fields[USER, ID], dataset.fields[ITEM, ID]

    # trainpipe
    trainpipe = freerec.data.postprocessing.source.RandomIDs(
        field=User, datasize=dataset.train().datasize
    ).sharding_filter().gen_train_uniform_sampling_(
        dataset, num_negatives=1
    ).batch(cfg.batch_size).column_().tensor_()

    validpipe = freerec.data.dataloader.load_gen_validpipe(
        dataset, batch_size=512, ranking=cfg.ranking
    )
    testpipe = freerec.data.dataloader.load_gen_testpipe(
        dataset, batch_size=512, ranking=cfg.ranking
    )

    tokenizer = FieldModuleList(dataset.fields)
    tokenizer.embed(
        cfg.embedding_dim, ID
    )
    model = BPRMF(tokenizer)

    graph = get_graph(cfg, dataset, NUM_PADS=0, itemonly=True)
    optimizer = get_optimizer(model, graph, cfg)

    criterion = freerec.criterions.BPRLoss()

    coach = CoachForBPRMF(
        trainpipe=trainpipe,
        validpipe=validpipe,
        testpipe=testpipe,
        fields=dataset.fields,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=None,
        device=cfg.device
    )
    coach.compile(
        cfg, 
        monitors=[
            'loss', 
            'hitrate@1', 'hitrate@5', 'hitrate@10',
            'ndcg@5', 'ndcg@10'
        ],
        which4best='ndcg@10'
    )
    graph.to(coach.device)
    coach.fit()


if __name__ == "__main__":
    main()