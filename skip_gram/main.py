from lib import data, models, criterions
import os
import torch.optim as optim
import torch

DATA_PATH = "/home/chris/projects/EntNet201119/tasks_1-20_v1-2/en"
FILE_NAME = "qa1_single-supporting-fact_train.txt"
SAVE_PATH = "/home/chris/projects/EntNet201119/docs/embedding"
NET_FILE = "checkpoint-Epoch-{}.data".format(6000)
SAVE_EPOCH = 1000
PRINT_EVERY = 500
LOAD_NET = True

BATCH_SIZE = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# preprocess the story from txt file
token_stories, answers, reasons, token_count, int2word, word2int, word_stories = data.preprocess_story(DATA_PATH, FILE_NAME)

# combined into one list
words = [word for story in token_stories for sentc in story for word in sentc]

# get the noise distribution in reversed order
noise_dist = torch.from_numpy(data.get_noise_dist(token_count, reversed=True))

# set up the model
embedding_dim = 64
model = models.SkipGramNeg(len(token_count), embedding_dim, noise_dist=noise_dist).to(DEVICE)

if LOAD_NET:
    print("Loading net params...")
    with open(os.path.join(SAVE_PATH, NET_FILE), "rb") as f:
        checkpoint = torch.load(f)
    model.load_state_dict(checkpoint['state_dict'])
    print("Successful!")

# using the loss that we defined
criterion = criterions.NegativeSamplingLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

steps, epoch = 0, 0
# train for some number of epochs
while True:
    for inputs, targets in data.get_batches(words, BATCH_SIZE, window_size=1):
        steps += 1
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

        # save the embedding layer
        if epoch % SAVE_EPOCH == 0 and epoch > 0:
            checkpoint = {"state_dict": model.state_dict()}
            with open(os.path.join(SAVE_PATH, "checkpoint-Epoch-{}.data".format(epoch)), "wb") as f:
                torch.save(checkpoint, f)

        # loss stats
        if steps % PRINT_EVERY == 0:
            print("Epoch: ", epoch)
            print("Loss: ", loss.item())  # avg batch loss at this point in training
            valid_examples, valid_similarities = criterions.cosine_similarity(model.in_embed, valid_size=16, valid_window=len(token_count), device=DEVICE)
            _, closest_idxs = valid_similarities.topk(6)

            valid_examples, closest_idxs = valid_examples.to('cpu'), closest_idxs.to('cpu')
            for ii, valid_idx in enumerate(valid_examples):
                closest_words = [int2word[idx.item()] for idx in closest_idxs[ii]][1:]
                print(int2word[valid_idx.item()] + " | " + ', '.join(closest_words))
            print("...\n")

    epoch += 1