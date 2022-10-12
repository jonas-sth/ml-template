from torch import nn


def xavier_weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
