
from collections import Mapping, Sequence
import numpy as np

from graphviz import Digraph

import torch
from torch.autograd import Variable
from torch.utils.data.dataloader import string_classes


def make_dot(var, params=None):
    """
    https://github.com/szagoruyko/functional-zoo/blob/master/visualize.py

    Produces Graphviz representation of PyTorch autograd graph
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '('+(', ').join(['%d' % v for v in size])+')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)
    add_nodes(var.grad_fn)
    return dot


def print_trainable_parameters(m):
    total_number = 0
    for name, p in m.named_parameters():
        print(name, p.size())
        total_number += np.prod(p.size())
    print("\nTotal number of trainable parameters: ", total_number)


def apply_variable(batch, **variable_kwargs):
    if torch.is_tensor(batch):
        return Variable(batch, **variable_kwargs)
    elif isinstance(batch, string_classes):
        return batch
    elif isinstance(batch, Mapping):
        return {k: apply_variable(sample, **variable_kwargs) for k, sample in batch.items()}
    elif isinstance(batch, Sequence):
        return [apply_variable(sample, **variable_kwargs) for sample in batch]
    else:
        raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                         .format(type(batch[0]))))