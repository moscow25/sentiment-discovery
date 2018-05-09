import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter

class LARC(object):
    def __init__(self, optimizer, trust_coefficient, epsilon=1e-8):
        self.param_groups = optimizer.param_groups
        self.optim = optimizer
        self.trust_coefficient = trust_coefficient
        self.eps = epsilon

    def __getstate__(self):
        return self.optim.__getstate__()

    def __setstate__(self, state):
        self.optim.__setstate__(state)

    def __repr__(self):
        return self.optim.__repr__()

    def state_dict(self):
        return self.optim.state_dict()

    def load_state_dict(self, state_dict):
        self.optim.load_state_dict(state_dict)

    def zero_grad(self):
        self.optim.zero_grad()

    def add_param_group(self, param_group):
        self.optim.add_param_group( param_group)

    def step(self):
        with torch.no_grad():
            for group in self.optim.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    param_norm = torch.norm(p)
                    adaptive_lr = (param_norm + self.eps) / (torch.norm(p.grad.data) + param_norm + self.eps)
                    p.grad.data *= self.trust_coefficient * adaptive_lr
        self.optim.step()
