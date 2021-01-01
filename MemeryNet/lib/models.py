import torch
from torch import nn
from codes.common_cmk import funcs
import torch.linalg as LA

class EntNet(nn.Module):
    def __init__(self, input_size, H_size, W_size, X_size, Y_size, Z_size, R_size, K_size, device):
        super(EntNet, self).__init__()
        self.H_size = H_size
        self.W_size = W_size
        self.device = device
        # dynamic memory
        self.H = nn.init.normal_(torch.empty(H_size, requires_grad=True, dtype=torch.float, device=self.device), mean=0.0, std=0.1)
        self.W = nn.init.normal_(torch.empty(W_size, requires_grad=True, dtype=torch.float, device=self.device), mean=0.0, std=0.1)

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

    def forward(self, E_s, Q, new_story=True):
        """
        k = sentence length
        m = memory size
        :param E_s: [ torch.tensor = facts in word embeddings ]
        :param Q: torch.tensor = Questions in word embeddings (n,PAD_MAX_LENGTH)
        :return: ans_vector (n,1)
        """
        if new_story:
            self.reset_memory()
        else:
            self.H = self.H.detach()
            self.W = self.W.detach()
        for E in E_s:
            # E = torch.tensor(data=E, requires_grad=True, dtype=torch.float)   # (64*k)
            s = torch.mul(self.params['F'], E).sum(dim=1).unsqueeze(1)  # (64*1)
            G = nn.Sigmoid()((torch.mm(s.t(), self.H) + torch.mm(s.t(), self.W)))  # (1*m)
            new_H = nn.Tanh()(torch.mm(self.params['X'], self.H) + torch.mm(self.params['Y'], self.W) + torch.mm(self.params['Z'], s))  # (64*m)
            self.H = funcs.unitVector_2d(self.H + torch.mul(G, new_H), dim=0)  # (64*m)

        # answer the question
        Q.requires_grad_()
        q = torch.mul(self.params['F'], Q).sum(dim=1).unsqueeze(1)  # (64*1)
        p = nn.Softmax(dim=1)(torch.mm(q.t(), self.H))  # (1*m)
        u = torch.mul(p, self.H).sum(dim=1).unsqueeze(1)  # (64*1)
        self.unit_params('R', dim=1)
        ans = torch.mm(self.params['R'], nn.Sigmoid()(q + torch.mm(self.params['K'], u)))  # (k,1)
        return ans

    # def answer(self, Q):
    #     """
    #     :param Q: torch.tensor = Questions in word embeddings (n,PAD_MAX_LENGTH)
    #     :return: ans_vector (n,1)
    #     """
    #     Q.requires_grad_()
    #     q = torch.mul(self.params['F'], Q).sum(dim=1).unsqueeze(1)    # (64*1)
    #     p = nn.Softmax(dim=1)(torch.mm(q.t(), self.params['H']))           # (1*m)
    #     u = torch.mul(p, self.params['H']).sum(dim=1).unsqueeze(1)    # (64*1)
    #     self.unit_params('R', dim=1)
    #     ans = torch.mm(self.params['R'], nn.Sigmoid()(q + torch.mm(self.params['K'], u))) # (k,1)
    #     # ans = nn.Softmax(dim=0)(y)
    #     return ans

    def reset_memory(self):
        self.H = nn.init.normal_(self.H).detach()
        self.W = nn.init.normal_(self.W).detach()

    def unit_params(self, name, dim):
        magnitude = self.params[name].data.detach().pow(2).sum(dim=dim).sqrt().unsqueeze(dim=dim)
        self.params[name].data = self.params[name] / magnitude