from lib import data, models, criterions
import torch
from torch import nn as nn
import torch.optim as optim
import os
from codes.common_cmk import funcs
import re
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# get the now time
now = datetime.now()
dt_string = now.strftime("%y%m%d_%H%M%S")
VERSION = 1

MAIN_PATH = "/home/chris/projects/201119_EntNet/docs/"
DATA_PATH = MAIN_PATH + "tasks_1-20_v1-2/en"
TRAIN_SET_NAME = "qa1_single-supporting-fact_train.txt"
TEST_SET_NAME = "qa1_single-supporting-fact_test.txt"

# for Embedding params
SAVE_EMBED_PATH = MAIN_PATH + str(VERSION) + "/embedding"
EMBED_FILE = "checkpoint-Epoch-{}-.data".format(6000)
INT2WORD = "int2word.txt"
WORD2INT = "word2int.txt"

# for EntNet params
SAVE_EntNET_PATH = MAIN_PATH + str(VERSION) + "/entNet_weights"
EntNET_FILE = "checkpoint-entNet-Epoch-{}.data".format(800)

# for run file to monitor progress in tensorboard
TENSORBOARD_SAVE_PATH = MAIN_PATH + str(VERSION) + "/runs/" + dt_string

DEVICE = "cuda"
SAVE_EPOCH = 50
TEST_EPOCH = 5
LOAD_NET = False

# Read the token_count, int2word, word2int
SkipGram_Net = data.load_file_from_SkipGram(SAVE_EMBED_PATH, EMBED_FILE, INT2WORD, WORD2INT)
Train = data.translate_story_into_token(DATA_PATH, TRAIN_SET_NAME, SkipGram_Net.word2int)
Test = data.translate_story_into_token(DATA_PATH, TEST_SET_NAME, SkipGram_Net.word2int)

# Load the model
EMBED_SIZE = SkipGram_Net.weights.size()[1] # 64
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
optimizer = optim.Adam(entNet.parameters(), lr=0.01)
criterion = criterions.Loss_Calculator()

writer = SummaryWriter(log_dir=TENSORBOARD_SAVE_PATH, comment="EntNet")
step = 0
while True:
    q_count, correct, losses = 0,0,0
    entNet.train()
    for E_s, Q, ans_vector, ans, new_story, end_story in data.generate_2(SkipGram_Net.embedding, Train.token_stories, Train.token_answers, SkipGram_Net.word2int,
                                                            fixed_length=PAD_MAX_LENGTH, device=DEVICE):
        predicted_vector = entNet.forward(E_s, Q, new_story=new_story)
        # predicted_vector = entNet.answer(Q)
        loss = criterion(predicted_vector, ans_vector)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses += loss.detach().cpu().item()

        # checking the similarity
        most_similarity_ans = funcs.get_most_similar_vectors_pos(SkipGram_Net.embedding_arr, predicted_vector.detach().cpu().numpy(), k=5)
        if most_similarity_ans[0] == ans:
            correct += 1

        # save the embedding layer
        if EPISODE % SAVE_EPOCH == 0:
            checkpoint = {"state_dict": entNet.state_dict()}
            with open(os.path.join(SAVE_EntNET_PATH, "checkpoint-entNet-Epoch-{}.data".format(EPISODE)), "wb") as f:
                torch.save(checkpoint, f)

        q_count += 1
        step += 1

    if EPISODE % TEST_EPOCH == 0:
        test_q_count, test_correct, test_losses = 0,0,0
        entNet.eval()
        for E_s, Q, ans_vector, ans, new_story, end_story in data.generate_2(SkipGram_Net.embedding, Test.token_stories, Test.token_answers, SkipGram_Net.word2int,
                                                                fixed_length=PAD_MAX_LENGTH, device=DEVICE):
            predicted_vector= entNet.forward(E_s, Q, new_story=new_story)
            # predicted_vector = entNet.answer(Q)
            loss = criterion(predicted_vector, ans_vector)
            test_losses += loss.detach().cpu().item()

            # checking the similarity
            most_similarity_ans = funcs.get_most_similar_vectors_pos(SkipGram_Net.embedding_arr, predicted_vector.detach().cpu().numpy(), k=5)
            if most_similarity_ans[0] == ans:
                test_correct += 1

            test_q_count += 1
        test_mean_loss = test_losses / test_q_count
        print("Test Mean Loss: {}".format(test_mean_loss))
        print("Test Accuracy: {:.3f}%".format(test_correct / test_q_count * 100))

    episode_mean_loss = losses / q_count
    print("Episode: {}; loss: {}".format(EPISODE, episode_mean_loss))
    print("Accuracy: {:.3f}%".format(correct / q_count * 100))
    writer.add_scalar("loss", episode_mean_loss, step)
    EPISODE += 1