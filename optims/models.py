

from typing import Callable

import torch
from freerec.models import RecSysArch
from freerec.data.tags import USER, ITEM, ID
from itertools import chain

from .utils import Smoother


class ArchWithFields(RecSysArch):

    def marked_params(self):
        raise NotImplementedError()

class UserItemArch(ArchWithFields):

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
            {
                'params': chain(User.parameters(), Item.parameters()), 
                'smoother': Smoother(graph, beta=cfg.beta3, L=cfg.L, aggr=cfg.aggr)
            },
            {'params': other},
        ]
        return params

class ItemOnlyArch(ArchWithFields):

    def marked_params(self, cfg, graph):
        other = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'fields' in name:
                continue
            else:
                other.append(param)
        Item = self.fields[ITEM, ID]
        params = [
            {
                'params': Item.parameters(),
                'smoother': Smoother(graph, beta=cfg.beta3, L=cfg.L, aggr=cfg.aggr)
            },
            {'params': other},
        ]
        return params


def get_optimizer(model: ArchWithFields, graph: Callable, cfg):
    from optims.AdamW import AdamWSEvo
    from optims.Adam import AdamSEvo
    from optims.SGD import SGDSEvo

    params = model.marked_params(cfg, graph)
    if cfg.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=cfg.lr, 
            momentum=cfg.momentum,
            nesterov=cfg.nesterov,
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer.lower() == 'sgdsevo':
        optimizer = SGDSEvo(
            params,
            lr=cfg.lr, momentum=cfg.momentum,
            nesterov=cfg.nesterov,
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer.lower() == 'adamsevo':
        optimizer = AdamSEvo(
            params,
            lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer.lower() == 'adamwsevo':
        optimizer = AdamWSEvo(
            params,
            lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay
        )
    return optimizer