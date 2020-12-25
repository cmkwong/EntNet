import torch
from torch import nn
import numpy as np
import torch.linalg as LA
import random
from lib import data, models, criterions
import torch
from torch import nn as nn
import torch.optim as optim
import os

class EntNet(nn.Module):
    def __init__(self, input_size, H_size, W_size, X_size, Y_size, Z_size, R_size, K_size, device):
        super(EntNet, self).__init__()
        self.H_size = H_size
        self.W_size = W_size
        self.device = device

        # embedding parameters
        self.params = nn.ParameterDict({
            'F': nn.Parameter(nn.init.normal_(torch.empty(input_size, requires_grad=True, dtype=torch.float), mean=0.0, std=0.1)),

            # dynamic memory
            'H': nn.Parameter(nn.init.normal_(torch.empty(H_size, requires_grad=True, dtype=torch.float), mean=0.0, std=0.1)),
            'W': nn.Parameter(nn.init.normal_(torch.empty(W_size, requires_grad=True, dtype=torch.float), mean=0.0, std=0.1)),

            # shared parameters
            'X': nn.Parameter(nn.init.normal_(torch.empty(X_size, requires_grad=True, dtype=torch.float), mean=0.0, std=0.1)),
            'Y': nn.Parameter(nn.init.normal_(torch.empty(Y_size, requires_grad=True, dtype=torch.float), mean=0.0, std=0.1)),
            'Z': nn.Parameter(nn.init.normal_(torch.empty(Z_size, requires_grad=True, dtype=torch.float), mean=0.0, std=0.1)),

            # answer parameters
            'R': nn.Parameter(nn.init.normal_(torch.empty(R_size, requires_grad=True, dtype=torch.float), mean=0.0, std=0.1)),
            'K': nn.Parameter(nn.init.normal_(torch.empty(K_size, requires_grad=True, dtype=torch.float), mean=0.0, std=0.1))
        }).to(self.device)

        # self.register_parameter('F', self.params['F'])
        # self.register_parameter('H', self.params['H'])
        # self.register_parameter('W', self.params['W'])
        # self.register_parameter('X', self.params['X'])
        # self.register_parameter('Y', self.params['Y'])
        # self.register_parameter('Z', self.params['Z'])
        # self.register_parameter('R', self.params['R'])
        # self.register_parameter('K', self.params['K'])

    def forward(self, E_s, new_story=True):
        """
        k = sentence length
        m = memory size
        :param E_s: [ torch.tensor = facts in word embeddings ]
        :return: Boolean
        """
        # if new_story:
        #     self.reset_memory()
        for E in E_s:
            # E = torch.tensor(data=E, requires_grad=True, dtype=torch.float)   # (64*k)
            s = torch.mul(self.params['F'], E).sum(dim=1).unsqueeze(1)  # (64*1)
            G = nn.Sigmoid()((torch.mm(s.t(), self.params['H']) + torch.mm(s.t(), self.params['W'])))  # (1*m)
            new_H = nn.Tanh()(
                torch.mm(self.params['X'], self.params['H']) + torch.mm(self.params['Y'], self.params['W']) + torch.mm(
                    self.params['Z'], s))  # (64*m)
            self.params['H'].data = self.params['H'] + torch.mul(G, new_H)  # (64*m)
            self.params['H'].data = self.params['H'] / LA.norm(self.params['H'], 2)  # (64*m)
        return True

    def answer(self, Q):
        """
        :param Q: torch.tensor = Questions in word embeddings (n,PAD_MAX_LENGTH)
        :return: ans_vector (n,1)
        """
        Q.requires_grad_()
        q = torch.mul(self.params['F'], Q).sum(dim=1).unsqueeze(1)  # (64*1)
        p = nn.Softmax(dim=1)(torch.mm(q.t(), self.params['H']))  # (1*m)
        u = torch.mul(p, self.params['H']).sum(dim=1).unsqueeze(1)  # (64*1)
        y = torch.mm(self.params['R'], nn.Sigmoid()(q + torch.mm(self.params['K'], u)))  # (k,1)
        ans = nn.Softmax(dim=0)(y)
        return ans

    def reset_memory(self):
        self.params['H'].data = nn.init.normal_(self.params['H']).to(self.device)
        self.params['W'].data = nn.init.normal_(self.params['W']).to(self.device)

DATA_PATH = "/home/chris/projects/201119_EntNet/tasks_1-20_v1-2/en"
FILE_NAME = "qa1_single-supporting-fact_train.txt"
SAVE_PATH = "/home/chris/projects/201119_EntNet/docs/embedding"
NET_FILE = "checkpoint-Epoch-{}.data".format(6000)
DEVICE = "cuda"
EPISODE = 1

# Load the story text file
token_stories, token_answers, token_reasons, int2word, word2int = data.preprocess_story(path=DATA_PATH, file_name=FILE_NAME)

# Load the embedding
print("Loading net params...")
with open(os.path.join(SAVE_PATH, NET_FILE), "rb") as f:
    checkpoint = torch.load(f)
weights = checkpoint['state_dict']['in_embed.weight']
embedding = nn.Embedding.from_pretrained(weights)
print("Successful!")

# Load the model
EMBED_SIZE = weights.size()[1] # 64
PAD_MAX_LENGTH = 10
M_SLOTS = 20
entNet = EntNet(input_size=(EMBED_SIZE,PAD_MAX_LENGTH),
                       H_size=(EMBED_SIZE,M_SLOTS),
                       W_size=(EMBED_SIZE,M_SLOTS),
                       X_size=(EMBED_SIZE,EMBED_SIZE),
                       Y_size=(EMBED_SIZE,EMBED_SIZE),
                       Z_size=(EMBED_SIZE,EMBED_SIZE),
                       R_size=(EMBED_SIZE,EMBED_SIZE),
                       K_size=(EMBED_SIZE,EMBED_SIZE),
                       device=DEVICE)
# optimizer
optimizer = optim.Adam(entNet.parameters(), lr=0.01)
criterion = criterions.Loss_Calculator()

while True:
    step = 1
    losses = 0
    for E_s, Q, ans, new_story in data.generate(embedding, token_stories, token_answers, word2int, fixed_length=PAD_MAX_LENGTH, device=DEVICE):
        entNet.forward(E_s, new_story=new_story)
        predicted_ans = entNet.answer(Q)
        # print(predicted_ans.size())
        loss = criterion(predicted_ans, ans)
        # optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses += loss.item()
        step += 1
    print("EPS - {} loss - {}".format(EPISODE, losses/step))
    EPISODE += 1