from lib import data, models, criterions
from codes.common_cmk import readFile
from codes.common_cmk.config import *
import os
import numpy as np
import torch.optim as optim
import torch
from torch.utils.tensorboard import SummaryWriter

# preprocess the story from txt file
token_stories, reasons, token_count, original_token_count, int2word, word2int, word_stories = data.preprocess_story(DATA_PATH, TRAIN_SET_NAME[TRAIN_DATA_INDEX])
# write the txt to store the word2int and int2word
readFile.write_dict(original_token_count, SAVE_EMBED_PATH, file_name=ORIGINAL_TOKEN_COUNT)
readFile.write_dict(token_count, SAVE_EMBED_PATH, file_name=TOKEN_COUNT)
readFile.write_dict(int2word, SAVE_EMBED_PATH, file_name=SAVED_INT2WORD)
readFile.write_dict(word2int, SAVE_EMBED_PATH, file_name=SAVED_WORD2INT)

# merge into one list
words = [word for story in token_stories for sentc in story for word in sentc]

# get the noise distribution in reversed order
noise_dist = torch.from_numpy(data.get_noise_dist(token_count, reversed=True))

# set up the model
model = models.SkipGramNeg(len(token_count), EMBED_SIZE, noise_dist=noise_dist).to(DEVICE)

if SG_LOAD_NET:
    print("Loading net params...")
    with open(os.path.join(SAVE_EMBED_PATH, EMBED_FILE), "rb") as f:
        checkpoint = torch.load(f)
    model.load_state_dict(checkpoint['state_dict'])
    print("Successful!")

# using the loss that we defined
criterion = criterions.NegativeSamplingLoss()
optimizer = optim.Adam(model.parameters(), lr=SG_LEARNING_RATE)

writer = SummaryWriter(log_dir=SG_TENSORBOARD_SAVE_PATH, comment="Skip_Gram")
epoch, steps = 0, 0
# train for some number of epochs
while True:
    episode_loss = []
    for inputs, targets in data.get_batches(words, SG_BATCH_SIZE, window_size=1):

        inputs, targets = torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        # input, output, and noise vectors
        input_vectors = model.forward_input(inputs)
        output_vectors = model.forward_output(targets)
        noise_vectors = model.forward_noise(inputs.shape[0], 2)

        # negative sampling loss
        loss = criterion(input_vectors, output_vectors, noise_vectors)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        episode_loss.append(loss.detach().cpu().item())

        # loss stats
        if steps % SG_PRINT_STEPS == 0 and steps > 0:
            print("Epoch: ", epoch)
            print("Loss: ", loss.detach().cpu().item())  # avg batch loss at this point in training
            valid_examples, valid_similarities = criterions.cosine_similarity(model.in_embed, valid_size=16, valid_window=len(token_count), device=DEVICE)
            _, closest_idxs = valid_similarities.topk(6)

            valid_examples, closest_idxs = valid_examples.to('cpu'), closest_idxs.to('cpu')
            for ii, valid_idx in enumerate(valid_examples):
                closest_words = [int2word[idx.item()] for idx in closest_idxs[ii]][1:]
                print(int2word[valid_idx.item()] + " | " + ', '.join(closest_words))
            print("...\n")

        steps += 1

    # save the embedding layer
    if epoch % SG_SAVE_EPOCH == 0 and epoch > 0:
        checkpoint = {"state_dict": model.state_dict()}
        with open(os.path.join(SAVE_EMBED_PATH, EMBED_FILE_SAVED.format(epoch)), "wb") as f:
            torch.save(checkpoint, f)

    if epoch % SG_WRITE_EPOCH == 0 and epoch > 0:
        results = {}
        valid_examples, valid_similarities = criterions.cosine_similarity(model.in_embed, valid_size=len(token_count), valid_window=len(token_count), device=DEVICE)
        _, closest_idxs = valid_similarities.topk(6)
        valid_examples, closest_idxs = valid_examples.to('cpu'), closest_idxs.to('cpu')
        for ii, valid_idx in enumerate(valid_examples):
            closest_words = [int2word[idx.item()] for idx in closest_idxs[ii]][1:]
            results[int2word[valid_idx.item()]] = closest_words
        readFile.write_dict(results, SAVE_EMBED_PATH, file_name=SG_RESULTS.format(epoch))

    # write to tensorboard
    writer.add_scalar('episode loss', np.mean(episode_loss), epoch)

    epoch += 1