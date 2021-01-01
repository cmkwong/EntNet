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
from codes.common_cmk import funcs


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
        E = E_s[0]
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

def generate(embedding, token_stories, token_answers, word2int, fixed_length=10, device="cuda"):
    """
    :param token_stories: [ [ [1,3,5,7,8,4,9,10,19],[1,3,5,7,8,4,9], [12,3,5,7,8,14,11], ... ], ... ]
    :param token_answers: [ [12,34], ... ]
    :return: [ torch.tensor(size=(n,fixed_length)), ... ], torch.tensor(size=(n,fixed_length)), torch_tensor(size=(n,1)), Boolean
    """
    for story_i, story in enumerate(token_stories):
        new_story = True
        Q_count = 0
        E_s = []
        story_len = len(story)
        for sentc_i, sentence in enumerate(story):
            if sentc_i == story_len - 1:
                end_story = True
            else:
                end_story = False
            E = data.sentc2e(embedding, sentence, fixed_length=fixed_length, device=device)
            if word2int['<q>'] in sentence:
                Q = E
                # acquire the ans
                ans = token_answers[story_i][Q_count]
                ans_vector = data.sentc2e(embedding, ans, fixed_length=1, device=device)
                yield E_s, Q, ans_vector, ans, new_story, end_story
                # reset after yield
                new_story = False
                E_s.clear()
                Q_count += 1
            else:
                E_s.append(E)

DATA_PATH = "/home/chris/projects/201119_EntNet/docs/tasks_1-20_v1-2/en"
FILE_NAME = "qa1_single-supporting-fact_train.txt"
SAVE_PATH = "/home/chris/projects/201119_EntNet/docs/1/embedding"
NET_FILE = "checkpoint-Epoch-{}.data".format(3000)
DEVICE = "cuda"
EPISODE = 1

# Load the story text file
Data = data.preprocess_story_Discard(path=DATA_PATH, file_name=FILE_NAME)

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
    for E_s, Q, ans_vector, ans, new_story, end_story in generate(embedding, Data.token_stories, Data.token_answers, Data.word2int, fixed_length=PAD_MAX_LENGTH, device=DEVICE):
        predicted_ans = entNet.forward(E_s, Q, new_story=new_story)
        # print(predicted_ans.size())
        loss = criterion(predicted_ans, ans_vector)
        optimizer.zero_grad()
        loss.backward(retain_graph=(not end_story))
        optimizer.step()
        losses += loss.item()
        step += 1
    print("EPS - {} loss - {}".format(EPISODE, losses/step))
    EPISODE += 1