from lib import data, models, criterions
import torch
from torch import nn as nn
import torch.optim as optim
import os
from codes.common_cmk import funcs
import re

DATA_PATH = "/home/chris/projects/201119_EntNet/tasks_1-20_v1-2/en"
FILE_NAME = "qa1_single-supporting-fact_train.txt"
# for Embedding
SAVE_EMBED_PATH = "/home/chris/projects/201119_EntNet/docs/embedding"
EMBED_FILE = "checkpoint-Epoch-{}.data".format(6000)
# for EntNet
SAVE_EntNET_PATH = "/home/chris/projects/201119_EntNet/docs/entNet_weights"
EntNET_FILE = "checkpoint-entNet-Epoch-{}.data".format(200)
DEVICE = "cuda"
SAVE_EPOCH = 10
LOAD_NET = True

# Load the embedding
print("Loading net params...")
with open(os.path.join(SAVE_EMBED_PATH, EMBED_FILE), "rb") as f:
    checkpoint = torch.load(f)
weights = checkpoint['state_dict']['in_embed.weight']
embedding = nn.Embedding.from_pretrained(weights)
embedding_arr = embedding.weight.data.cpu().detach().numpy()
print("Successful!")

# Load the story text file
token_stories, token_answers, token_reasons, int2word, word2int = data.preprocess_story(path=DATA_PATH, file_name=FILE_NAME)

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
if LOAD_NET:
    print("Loading net params...")
    with open(os.path.join(SAVE_EntNET_PATH, EntNET_FILE), "rb") as f:
        checkpoint = torch.load(f)
    entNet.load_state_dict(checkpoint['state_dict'])
    print("Successful!")
    EPISODE = int(re.findall('Epoch-(\d+)', EntNET_FILE)[0])
else:
    EPISODE = 1

# optimizer
optimizer = optim.Adam(entNet.parameters(), lr=0.1)
criterion = criterions.Loss_Calculator()

while True:
    step = 1
    losses = 0
    for E_s, Q, ans_vector, ans, new_story in data.generate(embedding, token_stories, token_answers, word2int, fixed_length=PAD_MAX_LENGTH, device=DEVICE):
        entNet.forward(E_s, new_story=new_story)
        predicted_vector = entNet.answer(Q)
        loss = criterion(predicted_vector, ans_vector)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses += loss.item()

        # checking the similarity
        most_similarity_ans = funcs.get_most_similar_vectors_pos(embedding_arr, predicted_vector.detach().cpu().numpy(), k=5)

        if step % 100 == 0:
            print("Predicted ans: {}, Answer: {}".format(most_similarity_ans, ans))

        # save the embedding layer
        if EPISODE % SAVE_EPOCH == 0:
            checkpoint = {"state_dict": entNet.state_dict()}
            with open(os.path.join(SAVE_EntNET_PATH, "checkpoint-entNet-Epoch-{}.data".format(EPISODE)), "wb") as f:
                torch.save(checkpoint, f)

        step += 1

    print("Episode: {}; loss: {}".format(EPISODE, losses/step))
    EPISODE += 1