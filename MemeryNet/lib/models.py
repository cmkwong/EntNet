import torch
from torch import nn
import numpy as np
import torch.linalg as LA
import random

class EntNet(nn.Module):
    def __init__(self, input_size, H_size, W_size, X_size, Y_size, Z_size, R_size, K_size):
        super(EntNet, self).__init__()
        self.H_size = H_size
        self.W_size = W_size

        # embedding parameters
        self.F = torch.tensor(data=np.random.normal(loc=0.0, scale=0.1, size=input_size), requires_grad=True, dtype=torch.float)

        # dynamic memory
        self.H = torch.tensor(data=np.random.normal(loc=0.0, scale=0.1, size=H_size), requires_grad=True, dtype=torch.float)
        self.W = torch.tensor(data=np.random.normal(loc=0.0, scale=0.1, size=W_size), requires_grad=True, dtype=torch.float)

        # shared parameters
        self.X = torch.tensor(data=np.random.normal(loc=0.0, scale=0.1, size=X_size), requires_grad=True, dtype=torch.float)
        self.Y = torch.tensor(data=np.random.normal(loc=0.0, scale=0.1, size=Y_size), requires_grad=True, dtype=torch.float)
        self.Z = torch.tensor(data=np.random.normal(loc=0.0, scale=0.1, size=Z_size), requires_grad=True, dtype=torch.float)

        # answer parameters
        self.R = torch.tensor(data=np.random.normal(loc=0.0, scale=0.1, size=R_size), requires_grad=True, dtype=torch.float)
        self.K = torch.tensor(data=np.random.normal(loc=0.0, scale=0.1, size=K_size), requires_grad=True, dtype=torch.float)

    def forward(self, E_s, new_story=True):
        """
        k = sentence length
        m = memory size
        :param E_s: [ torch.tensor = facts in word embeddings ]
        :return: Boolean
        """
        if new_story:
            self.reset_memory()
        for E in E_s:
            # E = torch.tensor(data=E, requires_grad=True, dtype=torch.float)                                 # (64*k)
            s = torch.mul(self.F, E).sum(dim=1).unsqueeze(1)                                                # (64*1)
            G = nn.Sigmoid()((torch.mm(s.t(), self.H) + torch.mm(s.t(), self.W)))                           # (1*m)
            new_H = nn.Tanh()(torch.mm(self.X, self.H) + torch.mm(self.Y, self.W) + torch.mm(self.Z, s))    # (64*m)
            self.H = self.H + torch.mul(G, new_H)                                                           # (64*m)
            self.H = self.H / LA.norm(self.H, 2)                                                            # (64*m)
        return True

    def answer(self, Q):
        """
        :param Q: torch.tensor = Questions in word embeddings (n,PAD_MAX_LENGTH)
        :return: ans_vector (n,1)
        """
        q = torch.mul(self.F, Q).sum(dim=1).unsqueeze(1)    # (64*1)
        p = nn.Softmax(dim=1)(torch.mm(q.t(), self.H))           # (1*m)
        u = torch.mul(p, self.H).sum(dim=1).unsqueeze(1)    # (64*1)
        y = torch.mm(self.R, nn.Sigmoid()(q + torch.mm(self.K, u))) # (k,1)
        ans = nn.Softmax(dim=0)(y)
        return ans

    def reset_memory(self):
        self.H = torch.tensor(data=np.random.normal(loc=0.0, scale=0.1, size=self.H_size), requires_grad=True, dtype=torch.float)
        self.W = torch.tensor(data=np.random.normal(loc=0.0, scale=0.1, size=self.W_size), requires_grad=True, dtype=torch.float)
