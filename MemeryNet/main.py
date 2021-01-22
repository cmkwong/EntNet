from lib import data, models, criterions
from codes.common_cmk.config import *
import torch
import torch.optim as optim
import os
from codes.common_cmk import funcs
import re
from torch.utils.tensorboard import SummaryWriter

# Read the token_count, int2word, word2int
SkipGram_Net = data.load_file_from_SkipGram(SAVE_EMBED_PATH, EMBED_FILE, INT2WORD, WORD2INT)
Train = data.translate_story_into_token(DATA_PATH, TRAIN_SET_NAME[TRAIN_DATA_INDEX], SkipGram_Net.word2int)
Test = data.translate_story_into_token(DATA_PATH, TEST_SET_NAME[TRAIN_DATA_INDEX], SkipGram_Net.word2int)

# Load the model
embed_size = SkipGram_Net.weights.size()[1] # 64
entNet = models.EntNet(input_size=(embed_size, PAD_MAX_LENGTH),
                       H_size=(embed_size, M_SLOTS),
                       W_size=(embed_size, M_SLOTS),
                       X_size=(embed_size, embed_size),
                       Y_size=(embed_size, embed_size),
                       Z_size=(embed_size, embed_size),
                       R_size=(embed_size, embed_size),
                       K_size=(embed_size, embed_size),
                       device=DEVICE)
if EntNet_LOAD_NET:
    print("Loading net params...", end=' ')
    with open(os.path.join(SAVE_EntNET_PATH, EntNET_FILE), "rb") as f:
        checkpoint = torch.load(f)
    entNet.load_state_dict(checkpoint['state_dict'])
    print("Successful!")
    episode = int(re.findall('Epoch-(\d+)', EntNET_FILE)[0])
else:
    episode = 1

# optimizer
optimizer = optim.Adam(entNet.parameters(), lr=EntNET_LEARNING_RATE)
criterion = criterions.Loss_Calculator()

writer = SummaryWriter(log_dir=EntNet_TENSORBOARD_SAVE_PATH, comment="EntNet")
with data.Episode_Tracker(SkipGram_Net.int2word, RESULT_CHECKING_PATH, writer, episode=episode, write_episode=5) as tracker:
    while True:
        entNet.train()
        q_count, correct, losses = 0, 0, 0
        for T in data.generate(SkipGram_Net.embedding, Train.token_stories, Train.token_answers, SkipGram_Net.word2int,
                                                                fixed_length=PAD_MAX_LENGTH, device=DEVICE):
            entNet.forward(T.E_s, new_story=T.new_story)
            predicted_vector = entNet.answer(T.Q)
            loss = criterion(predicted_vector, T.ans_vector)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.detach().cpu().item()

            # checking the similarity
            most_similarity_ans = funcs.get_most_similar_vectors_pos(SkipGram_Net.embedding_arr, predicted_vector.detach().cpu().numpy(), k=5)
            if most_similarity_ans[0] == T.ans:
                correct += 1

            # print the story for inspect
            if tracker.episode % EntNet_TEST_EPOCH == 0:
                tracker.write(T.stories, T.q, most_similarity_ans[0], T.ans, T.new_story, T.end_story, "Train")

            # save the embedding layer
            if tracker.episode % EntNet_SAVE_EPOCH == 0:
                checkpoint = {"state_dict": entNet.state_dict()}
                with open(os.path.join(SAVE_EntNET_PATH, EntNET_FILE_FORMAT.format(TRAIN_DATA_INDEX, VERSION, tracker.episode)), "wb") as f:
                    torch.save(checkpoint, f)

            q_count += 1

        if tracker.episode % EntNet_TEST_EPOCH == 0:
            test_q_count, test_correct, test_losses = 0,0,0
            entNet.eval()
            for t in data.generate(SkipGram_Net.embedding, Test.token_stories, Test.token_answers, SkipGram_Net.word2int,
                                                                    fixed_length=PAD_MAX_LENGTH, device=DEVICE):
                entNet.forward(t.E_s, new_story=t.new_story)
                predicted_vector = entNet.answer(t.Q)
                loss = criterion(predicted_vector, t.ans_vector)
                test_losses += loss.detach().cpu().item()

                # checking the similarity
                most_similarity_ans = funcs.get_most_similar_vectors_pos(SkipGram_Net.embedding_arr, predicted_vector.detach().cpu().numpy(), k=5)
                if most_similarity_ans[0] == t.ans:
                    test_correct += 1

                tracker.write(t.stories, t.q, most_similarity_ans[0], t.ans, t.new_story, t.end_story, "Test")

                test_q_count += 1

            tracker.print_episode_status(test_q_count, test_correct, test_losses, "Test")
            tracker.plot_episode_status(test_q_count, test_correct, test_losses, "Test")

        tracker.print_episode_status(q_count, correct, losses, "Train")
        tracker.plot_episode_status(q_count, correct, losses, "Train")
        tracker.episode += 1