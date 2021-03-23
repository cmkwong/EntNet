import torch
import numpy as np
from torch import nn
from common_cmk import funcs
from MemeryNet.lib import descriptor
import pickle

class EntNet(nn.Module):

    def __init__(self, W, embed_size, m_slots, sentc_max_len, device):
        super(EntNet, self).__init__()
        self.device = device
        # dynamic memory
        self.H = torch.zeros((m_slots, embed_size), dtype=torch.float, device=self.device)
        self.W = W.clone().detach()

        # learnable activation layer
        # self.activation = nn.LeakyReLU(negative_slope=0.01).to(self.device)

        # embedding parameters
        self.params = nn.ParameterDict({
            'F': nn.Parameter(nn.init.normal_(torch.empty((sentc_max_len, embed_size), requires_grad=True, dtype=torch.float, device=self.device), mean=0.0, std=0.1)),

            # shared parameters
            'X': nn.Parameter(nn.init.normal_(torch.empty((embed_size, embed_size), requires_grad=True, dtype=torch.float, device=self.device), mean=0.0, std=0.1)),
            'Y': nn.Parameter(nn.init.normal_(torch.empty((embed_size, embed_size), requires_grad=True, dtype=torch.float, device=self.device), mean=0.0, std=0.1)),
            'Z': nn.Parameter(nn.init.normal_(torch.empty((embed_size, embed_size), requires_grad=True, dtype=torch.float, device=self.device), mean=0.0, std=0.1)),
            'X_b': nn.Parameter(torch.zeros((m_slots, embed_size), requires_grad=True, dtype=torch.float, device=self.device)),
            'Y_b': nn.Parameter(torch.zeros((m_slots, embed_size), requires_grad=True, dtype=torch.float, device=self.device)),
            'Z_b': nn.Parameter(torch.zeros((1, embed_size), requires_grad=True, dtype=torch.float, device=self.device)),

            # answer parameters
            'D': nn.Parameter(nn.init.normal_(torch.empty((sentc_max_len, embed_size), requires_grad=True, dtype=torch.float, device=self.device), mean=0.0, std=0.1)),
            'R': nn.Parameter(nn.init.normal_(torch.empty((m_slots, embed_size), requires_grad=True, dtype=torch.float, device=self.device), mean=0.0, std=0.1)),
            'K': nn.Parameter(nn.init.normal_(torch.empty((embed_size, embed_size), requires_grad=True, dtype=torch.float, device=self.device), mean=0.0, std=0.1)),
            'R_b': nn.Parameter(torch.zeros((m_slots, 1), requires_grad=True, dtype=torch.float, device=self.device)),
            'K_b': nn.Parameter(torch.zeros((1, embed_size), requires_grad=True, dtype=torch.float, device=self.device))
        })

        # dropout
        # self.dropout = nn.Dropout(p=0.3)

    def forward(self, E_s, new_story=True):
        """
        k = sentence length
        m = memory size
        :param E_s: [ torch.tensor = facts in word embeddings ]
        :return: ans_vector (n,1)
        """
        self.prepare_memory(new_story)
        for E in E_s:
            # E = torch.tensor(data=E, requires_grad=True, dtype=torch.float)   # (64*k)
            self.s = torch.mul(self.params['F'], E).sum(dim=0).unsqueeze(0)  # (1*64)
            self.G = nn.Sigmoid()((torch.mm(self.H, self.s.t()) + torch.mm(self.W, self.s.t())))  # (m*1)
            self.new_H = nn.Sigmoid()(   torch.addmm(self.params['X_b'], self.H, self.params['X']) +
                                            torch.addmm(self.params['Y_b'], self.W, self.params['Y']) +
                                            torch.addmm(self.params['Z_b'], self.s, self.params['Z']))  # (m*64)
            # self.H = funcs.unitVector_2d(self.H + torch.mul(self.G, self.new_H), dim=1)  # (m*64)
            self.H = 0.4 * self.H + 0.6 * torch.mul(self.G, self.new_H)

    def answer(self, Q):
        """
        :param Q: torch.tensor = Questions in word embeddings (n,PAD_MAX_LENGTH)
        :return: ans_vector (n,1)
        """
        # answer the question
        self.q = torch.mul(self.params['D'], Q).sum(dim=0).unsqueeze(0)  # (1*64)
        self.p = nn.Softmax(dim=0)(torch.mm(self.H, self.q.t()))  # (m*1)
        self.u = torch.mul(self.p, self.H).sum(dim=0).unsqueeze(0)  # (1*64)
        # self.unit_params('R', dim=1)
        self.ans_vector = torch.addmm(
            self.params['R_b'], self.params['R'], nn.Sigmoid()(self.q + torch.addmm(
                self.params['K_b'], self.u, self.params['K'])).t()
        )
        self.ans = nn.LogSoftmax(dim=1)(self.ans_vector.t())
        return self.ans

    def run_model(self, dataset, criterion, optimizer, device, mode="Train"):
        """
        :param dataset: collections.namedtuple('DataSet', ["E_s", 'Q', "ans_vector", "ans", "new_story", "end_story", 'stories', 'q'])
        :param criterion: criterion
        :param optimizer: optimizer
        :param mode: string: train/test
        :return: detached_loss, predict_ans
        """
        if mode == "Train":
            self.train()
        elif mode == "Test":
            self.train()
        self.forward(dataset.E_s, new_story=dataset.new_story)
        predict = self.answer(dataset.Q)
        loss = criterion(predict, torch.tensor([dataset.ans], device=device))
        if mode == "Train":
            if dataset.end_story:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            else:
                loss.backward(retain_graph=True)
        # detach the loss and predicted vector
        detached_loss = loss.detach().cpu().item()
        predict_ans = torch.argmax(predict.detach().cpu()).item() # get the ans value in integer
        return detached_loss, predict_ans

    def prepare_memory(self, new_story):
        if new_story:
            self.H = torch.zeros(self.H.shape, dtype=torch.float, device=self.device)

    def unit_params(self, name, dim):
        magnitude = self.params[name].data.detach().pow(2).sum(dim=dim).sqrt().unsqueeze(dim=dim)
        self.params[name].data = self.params[name] / magnitude