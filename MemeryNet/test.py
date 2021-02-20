import torch
import numpy as np
from torch import nn
from codes.common_cmk import funcs
from codes.MemeryNet.lib import descriptor
import pickle

class EntNet(nn.Module):

    def __init__(self, W, input_size, H_size, X_size, Y_size, Z_size, R_size, K_size, device):

        super(EntNet, self).__init__()
        self.record_allowed = False
        self.H_size = H_size
        self.device = device
        # dynamic memory
        self.H = nn.init.normal_(torch.empty(H_size, dtype=torch.float, device=self.device), mean=0.0, std=0.1)
        self.W = W.clone().detach()

        # embedding parameters
        self.params = nn.ParameterDict({
            'F': nn.Parameter(nn.init.normal_(torch.empty(input_size, requires_grad=True, dtype=torch.float, device=self.device), mean=0.0, std=0.1)),

            # shared parameters
            'X': nn.Parameter(nn.init.normal_(torch.empty(X_size, requires_grad=True, dtype=torch.float, device=self.device), mean=0.0, std=0.1)),
            'Y': nn.Parameter(nn.init.normal_(torch.empty(Y_size, requires_grad=True, dtype=torch.float, device=self.device), mean=0.0, std=0.1)),
            'Z': nn.Parameter(nn.init.normal_(torch.empty(Z_size, requires_grad=True, dtype=torch.float, device=self.device), mean=0.0, std=0.1)),

            # answer parameters
            'R': nn.Parameter(nn.init.normal_(torch.empty(R_size, requires_grad=True, dtype=torch.float, device=self.device), mean=0.0, std=0.1)),
            'K': nn.Parameter(nn.init.normal_(torch.empty(K_size, requires_grad=True, dtype=torch.float, device=self.device), mean=0.0, std=0.1))
        })

        # dropout
        self.dropout = nn.Dropout(p=0.3)

W = nn.init.normal_(torch.empty((64,36), dtype=torch.float, device="cuda"), mean=0.0, std=0.1)
net = EntNet(W, input_size=(64,64), H_size=(64,64), X_size=(64,64), Y_size=(64,64), Z_size=(64,64), R_size=(64,64), K_size=(64,64), device="cuda")