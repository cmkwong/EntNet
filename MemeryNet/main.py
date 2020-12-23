from lib import data, models, criterions
import torch
from torch import nn as nn
import torch.optim as optim
import os

DATA_PATH = "/home/chris/projects/EntNet201119/tasks_1-20_v1-2/en"
FILE_NAME = "qa1_single-supporting-fact_train.txt"
SAVE_PATH = "/home/chris/projects/EntNet201119/docs/embedding"
NET_FILE = "checkpoint-Epoch-{}.data".format(6000)
DEVICE = "cuda"

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
entNet = models.EntNet(input_size=(EMBED_SIZE,PAD_MAX_LENGTH),
                       H_size=(EMBED_SIZE,M_SLOTS), 
                       W_size=(EMBED_SIZE,M_SLOTS), 
                       X_size=(EMBED_SIZE,EMBED_SIZE), 
                       Y_size=(EMBED_SIZE,EMBED_SIZE), 
                       Z_size=(EMBED_SIZE,EMBED_SIZE), 
                       R_size=(EMBED_SIZE,EMBED_SIZE), 
                       K_size=(EMBED_SIZE,EMBED_SIZE),
                       device=DEVICE)
# optimizer
optimizer = optim.Adam(entNet.parameters(), lr=0.00001)

for E_s, Q, ans, new_story in data.generate(embedding, token_stories, token_answers, word2int, fixed_length=PAD_MAX_LENGTH, device=DEVICE):
    entNet.forward(E_s, new_story=new_story)
    predicted_ans = entNet.answer(Q)
    print(predicted_ans.size())
