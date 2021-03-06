from lib import data, models, criterions, graph
from codes.common_cmk.config import *
import torch
import torch.optim as optim
import os
import re
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Read the token_count, int2word, word2int
SkipGram_Net = data.load_file_from_SkipGram(SAVE_EMBED_PATH, EMBED_FILE, LOADED_INT2WORD, LOADED_WORD2INT)
Train = data.translate_story_into_token(DATA_PATH, TRAIN_SET_NAME[TRAIN_DATA_INDEX], SkipGram_Net.word2int)
Test = data.translate_story_into_token(DATA_PATH, TEST_SET_NAME[TRAIN_DATA_INDEX], SkipGram_Net.word2int)

# Load the model
embed_size = SkipGram_Net.weights.size()[1] # 64
M_SLOTS = SkipGram_Net.weights.t().size()[1]
entNet = models.EntNet( W=SkipGram_Net.weights.t(),
                        input_size=(embed_size, PAD_MAX_LENGTH),
                        H_size=(embed_size, M_SLOTS),
                        X_size=(embed_size, embed_size),
                        Y_size=(embed_size, embed_size),
                        Z_size=(embed_size, embed_size),
                        R_size=(M_SLOTS, embed_size),
                        K_size=(embed_size, embed_size),
                        device=DEVICE)
# generate training and test set
train_set = data.generate_data(SkipGram_Net.embedding, Train.token_stories, Train.token_answers, SkipGram_Net.word2int,
                               fixed_length=PAD_MAX_LENGTH, device=DEVICE)
test_set = data.generate_data(SkipGram_Net.embedding, Test.token_stories, Test.token_answers, SkipGram_Net.word2int,
                               fixed_length=PAD_MAX_LENGTH, device=DEVICE)

if EntNet_LOAD_NET:
    print("Loading net params: {}...".format(EntNET_FILE), end=' ')
    with open(os.path.join(SAVE_EntNET_PATH, EntNET_FILE), "rb") as f:
        checkpoint = torch.load(f)
    entNet.load_state_dict(checkpoint['state_dict'])
    print("Successful!")
    episode = int(re.findall('Epoch(\d+)', EntNET_FILE)[0])
else:
    episode = 1
    if EntNet_LOAD_INIT:
        print("Loading init net params: {}...".format(EntNET_INIT_FILE), end=' ')
        with open(os.path.join(SAVE_EntNET_PATH, EntNET_INIT_FILE), "rb") as f:
            checkpoint = torch.load(f)
        entNet.load_state_dict(checkpoint['state_dict'])
        print("Successful!")
    else:
        checkpoint = {"state_dict": entNet.state_dict()}
        with open(os.path.join(SAVE_EntNET_PATH, EntNET_INIT_FILE_SAVED), "wb") as f:
            torch.save(checkpoint, f)

# optimizer
optimizer = optim.Adam(entNet.parameters(), lr=EntNET_LEARNING_RATE)
criterion = criterions.NLLLoss()

writer = SummaryWriter(log_dir=EntNet_TENSORBOARD_SAVE_PATH, comment="EntNet")
with data.Episode_Tracker(entNet, SkipGram_Net.int2word, RESULT_CHECKING_PATH, writer, episode=episode, write_episode=5) as tracker:
    while True:

        q_count, correct, losses = 0, 0, 0
        # record the state before backpropagation
        if tracker.episode % EntNet_STATE_EPOCH == 0:
            entNet.record_allowed = True
        else:
            entNet.record_allowed = False

        # shuffle the story
        shuffle_num = np.arange(len(train_set))
        if SHUFFLE_TRAIN:
            np.random.shuffle(shuffle_num)

        # training
        for index in shuffle_num:
            story = train_set[index]
            for T in story:

                # run the model
                loss, predict_ans = entNet.run_model(T, criterion, optimizer, DEVICE, mode="Train")
                losses += loss

                # checking correction
                if predict_ans == T.ans:
                    correct += 1

                # print the story for inspect
                if tracker.episode % EntNet_RESULT_EPOCH == 0:
                    tracker.write(T.stories, T.q, predict_ans, T.ans, T.new_story, T.end_story, "Train")

                q_count += 1

        # testing
        if tracker.episode % EntNet_TEST_EPOCH == 0:
            # init
            test_q_count, test_correct, test_losses = 0,0,0
            entNet.record_allowed = False
            # shuffle
            shuffle_num = np.arange(len(test_set))
            if SHUFFLE_TRAIN:
                np.random.shuffle(shuffle_num)
            for index in shuffle_num:
                story = test_set[index]
                for t in story:

                    loss, predict_ans = entNet.run_model(t, criterion, None, DEVICE, mode="Test")
                    test_losses += loss

                    # checking correction
                    if predict_ans == t.ans:
                        test_correct += 1

                    tracker.write(t.stories, t.q, predict_ans, t.ans, t.new_story, t.end_story, "Test")

                    test_q_count += 1

            tracker.print_episode_status(test_q_count, test_correct, test_losses, "Test")
            tracker.plot_episode_status(test_q_count, test_correct, test_losses, "Test")

        # save the entNet layer
        if tracker.episode % EntNet_SAVE_EPOCH == 0:
            checkpoint = {"state_dict": entNet.state_dict()}
            with open(os.path.join(SAVE_EntNET_PATH, EntNET_FILE_SAVED.format(tracker.episode)), "wb") as f:
                torch.save(checkpoint, f)

        if tracker.episode % WEIGHT_HIST_EPOCH == 0:
            tracker.weight_histogram()

        if tracker.episode % WEIGHT_IMAGE_EPOCH == 0 and False:
            tracker.weight_image()

        tracker.print_episode_status(q_count, correct, losses, "Train")
        tracker.plot_episode_status(q_count, correct, losses, "Train")
        if tracker.episode % EntNet_STATE_EPOCH == 0: entNet.snapshot(STATE_CHECKING_PATH, STATE_MATRICS, PARAMS_MATRICS, GRADIEND_MATRICS,
                                                                       STATE_PATH, tracker.episode)

        tracker.episode += 1