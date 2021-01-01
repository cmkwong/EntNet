import torch
import torch.nn as nn
from collections import Counter
from codes.common_cmk import readFile, funcs
import collections
import os

def sentc2e(embedding, sentc_ints, fixed_length=6, device="cuda"):
    """
    :param embedding: nn.Embedding
    :param sentc_ints: [int] / int
    :return:  torch size = (64, max_length)
    """
    # cat the sentcs in fixed length
    if isinstance(sentc_ints, int):
        sentc_ints = [sentc_ints]
    sentc_ints.extend([sentc_ints[-1]] * (fixed_length - len(sentc_ints)))
    embed_vectors = None
    sentc_ints = torch.tensor(sentc_ints, dtype=torch.long).to(device)
    for i, sentc_int in enumerate(sentc_ints):
        if i == 0:
            embed_vectors = embedding(sentc_int).unsqueeze(0).t()
        else:
            embed_vectors = torch.cat((embed_vectors, embedding(sentc_int).unsqueeze(0).t()), dim=1)
    return embed_vectors
# embedding = torch.nn.Embedding(21, 64)
# sentc_ints = [20,4,1,2]
# e = sentc2e(embedding, sentc_ints, 10)

def preprocess_story_Discard(path, file_name):

    Data = collections.namedtuple("Data", ["token_stories", "token_answers", "reasons", "int2word", "word2int"])

    # read file
    raw_stories, answers, reasons = readFile.read_story(path, file_name)

    # label special charactor
    stories = []
    for story in raw_stories:
        stories.append([readFile.label_special_token(sentc) for sentc in story])

    # tokenize into word from the sentence
    word_stories = [readFile.tokenize_sentences(story) for story in stories]

    token_count = Counter()
    for word_story in word_stories:
        tc = readFile.get_token_count_from_sentences(word_story)
        token_count = token_count + tc

    # create word2int and int2word
    Data.int2word, Data.word2int = readFile.create_lookup_table(token_count, reverse=True)

    # label the word as int
    Data.token_stories = [readFile.label_word_as_int_token(word_story, Data.word2int) for word_story in word_stories]
    Data.token_answers = readFile.label_word_as_int_token(answers, Data.word2int)

    # separate the facts and questions
    # token_facts, token_questions = [], []
    # for token_story in token_stories:
    #     token_fact, token_question = readFile.target_detach(token_story, target_label=word2int['<q>'])
    #     token_facts.append(token_fact)
    #     token_questions.append(token_question)
    return Data

def load_file_from_SkipGram(path, embedding_file, int2word_file, word2int_file):

    SkipGram_Net = collections.namedtuple("SkipGram_Net", ["weights", "embedding", "embedding_arr", "token_count", "int2word", "word2int"])

    # Load the embedding
    print("Loading net params ...")
    with open(os.path.join(path, embedding_file), "rb") as f:
        checkpoint = torch.load(f)
    weights = checkpoint['state_dict']['in_embed.weight']
    SkipGram_Net.weights = funcs.unitVector_2d(weights, dim=0) # normalize into unit vector
    SkipGram_Net.embedding = nn.Embedding.from_pretrained(SkipGram_Net.weights)
    SkipGram_Net.embedding_arr = SkipGram_Net.embedding.weight.data.cpu().detach().numpy()
    print("Successful!")

    # load the int2word and word2int files
    print("Loading word2int and int2word files ...")
    SkipGram_Net.int2word = readFile.read_dict(path, int2word_file)
    SkipGram_Net.word2int = readFile.read_dict(path, word2int_file)
    print("Successful!")

    return SkipGram_Net

def translate_story_into_token(path, file_name, word2int):

    Data = collections.namedtuple("Train_Data", ["token_stories", "token_answers", "reasons"])

    # read file
    raw_stories, answers, Data.reasons = readFile.read_story(path, file_name)

    # label special charactor
    stories = []
    for story in raw_stories:
        stories.append([readFile.label_special_token(sentc) for sentc in story])

    # tokenize into word from the sentence
    word_stories = [readFile.tokenize_sentences(story) for story in stories]

    # label the word as int
    Data.token_stories = [readFile.label_word_as_int_token(word_story, word2int) for word_story in word_stories]
    Data.token_answers = readFile.label_word_as_int_token(answers, word2int)

    return Data

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
        for sentence in story:
            E = sentc2e(embedding, sentence, fixed_length=fixed_length, device=device)
            if word2int['<q>'] in sentence:
                Q = E
                # acquire the ans
                ans = token_answers[story_i][Q_count]
                ans_vector = sentc2e(embedding, ans, fixed_length=1, device=device)
                yield E_s, Q, ans_vector, ans, new_story
                # reset after yield
                new_story = False
                E_s.clear()
                Q_count += 1
            else:
                E_s.append(E)

def generate_2(embedding, token_stories, token_answers, word2int, fixed_length=10, device="cuda"):
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
            E = sentc2e(embedding, sentence, fixed_length=fixed_length, device=device)
            if word2int['<q>'] in sentence:
                Q = E
                # acquire the ans
                ans = token_answers[story_i][Q_count]
                ans_vector = sentc2e(embedding, ans, fixed_length=1, device=device)
                yield E_s, Q, ans_vector, ans, new_story, end_story
                # reset after yield
                new_story = False
                E_s.clear()
                Q_count += 1
            else:
                E_s.append(E)