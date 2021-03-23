import torch
import torch.nn as nn
import numpy as np
from common_cmk.config import *
from common_cmk import readFile, funcs
import collections
import os

def cat_zeros(embedding, max_length, dim, device="cuda"):
    """
    :param embedding: 2d tensor
    :param max_length: int
    :param dim: int
    :return:
    """
    assert (max_length - embedding.shape[dim] >= 0, "max sentence length should be larger or equal than padding length")
    zeros = None
    length = max_length - embedding.shape[dim]
    # create zero tensor, size depend on the dim required
    if dim == 0:
        zeros = torch.zeros((length, embedding.shape[1])).to(device)
    elif dim == 1:
        zeros = torch.zeros((embedding.shape[0], length)).to(device)
    required_embedding = torch.cat((embedding, zeros), dim=dim)
    return required_embedding

def sentc2e(embedding, sentc_ints, device="cuda"):
    """
    :param embedding: nn.Embedding
    :param sentc_ints: [int] / int
    :return:  torch size = (64, max_length)
    """
    # cat the sentcs in fixed length
    if isinstance(sentc_ints, int):
        sentc_ints = [sentc_ints]
    # sentc_ints.extend([0] * (fixed_length - len(sentc_ints)))
    embed_vectors = None
    sentc_ints = torch.tensor(sentc_ints, dtype=torch.long).to(device)
    for i, sentc_int in enumerate(sentc_ints):
        if i == 0:
            embed_vectors = embedding(sentc_int).unsqueeze(0)
        else:
            embed_vectors = torch.cat((embed_vectors, embedding(sentc_int).unsqueeze(0)), dim=0)
    # fixed_len_embed_vectors = fill_zeros(embed_vectors, fixed_length, dim=0, device=device)
    return embed_vectors

def load_file_from_SkipGram(path, embedding_file, int2word_file, word2int_file):

    SkipGram_Net = collections.namedtuple("SkipGram_Net", ["weights", "embedding", "embedding_arr", "token_count", "int2word", "word2int"])

    # Load the embedding
    print("Loading Embedding: {} ...".format(embedding_file), end=' ')
    with open(os.path.join(path, embedding_file), "rb") as f:
        checkpoint = torch.load(f)
    weights = checkpoint['state_dict']['in_embed.weight']
    SkipGram_Net.weights = funcs.unitVector_2d(weights, dim=1)                              # normalize into unit tensor
    SkipGram_Net.embedding = nn.Embedding.from_pretrained(SkipGram_Net.weights)             # pretrained embedding network
    SkipGram_Net.embedding_arr = SkipGram_Net.embedding.weight.data.cpu().detach().numpy()  # embedding array
    print("Successful!")

    # load the int2word and word2int files
    print("Loading {} and {} ...".format(word2int_file, int2word_file),end=' ')
    SkipGram_Net.int2word = readFile.read_dict(path, int2word_file)
    SkipGram_Net.word2int = readFile.read_dict(path, word2int_file)
    print("Successful!")

    return SkipGram_Net

def get_summary(stories):
    """
    :param stories: dict: {0: [Q1 part, Q2 part, ... ], 1: [Q1 part, Q2 part, ... ], ... }
    :return: dict: {max_sentc_len, max_q_num, min_q_num, max_session_len: int, int, int, int}
    """
    stat = {"story_len": 0, "max_sentc_len":0, "max_q_num":0, "min_q_num":float('inf'), "max_session_len":0}
    stat["story_len"] = len(stories)
    for story in stories.values():
        if len(story) > stat["max_q_num"]:
            stat["max_q_num"] = len(story)
        if len(story) < stat["min_q_num"]:
            stat["min_q_num"] = len(story)
        for session in story:
            if len(session.E_s) > stat["max_session_len"]:
                stat["max_session_len"] = len(session.E_s)
            for sentc in session.E_s:
                if len(sentc) > stat["max_sentc_len"]:
                    stat["max_sentc_len"] = len(sentc)
    return stat

def translate_story_into_token(path, file_name, word2int):

    Data = collections.namedtuple("Data", ["token_stories", "token_answers", "reasons"])

    # read file
    raw_stories, answers, Data.reasons = readFile.read_as_story(path, file_name)

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

def generate(embedding, token_stories, token_answers, word2int, device="cuda"):
    """
    :param token_stories: [ [ [1,3,5,7,8,4,9,10,19],[1,3,5,7,8,4,9], [12,3,5,7,8,14,11], ... ], ... ]
    :param token_answers: [ [12,34], ... ]
    :param word2int: dict
    :return: GenSet
    """
    GenSet = collections.namedtuple('GenSet', ["E_s", 'Q', "ans_vector", "ans", "new_story", "end_story", 'stories', 'q'])

    for story_i, story in enumerate(token_stories):
        GenSet.new_story = True
        Q_count = 0
        GenSet.E_s, GenSet.stories = [], []
        story_len = len(story)
        for sentc_i, sentence in enumerate(story):
            # check if this sentence is the last one, if so, set the end_story = True
            if sentc_i == story_len - 1:
                GenSet.end_story = True
            else:
                GenSet.end_story = False
            E = sentc2e(embedding, sentence, device=device)
            if word2int['<q>'] in sentence:
                GenSet.Q = E
                GenSet.q = sentence
                # acquire the ans
                GenSet.ans = token_answers[story_i][Q_count]
                GenSet.ans_vector = sentc2e(embedding, GenSet.ans, device=device)
                yield GenSet
                # reset after yield
                GenSet.new_story = False
                GenSet.E_s.clear()
                Q_count += 1
            else:
                GenSet.E_s.append(E)
                GenSet.stories.append(sentence)

def generate_data(embedding, token_stories, token_answers, word2int, device="cuda"):
    """
    :param token_stories: [ [ [1,3,5,7,8,4,9,10,19],[1,3,5,7,8,4,9], [12,3,5,7,8,14,11], ... ], ... ]
    :param token_answers: [ [12,34], ... ]
    :param word2int: dict
    :return: {story_i: [ nametuple:
                            "E_s": [embedding, ...],
                            'Q': embedding,
                            "ans_vector": embedding,
                            "ans": int,
                            "new_story": Boolean,
                            "end_story": Boolean,
                            'stories': [int],
                            'q': [int]
                        ], ...
                }
    data: {"E_s", 'Q', "ans_vector", "ans", "new_story", "end_story", 'stories', 'q'}
    """

    def init_data():
        DataSet = collections.namedtuple('DataSet', ["E_s", 'Q', "ans_vector", "ans", "new_story", "end_story", 'stories', 'q'])
        DataSet.E_s, DataSet.stories = [], []
        DataSet.Q, DataSet.q, DataSet.ans_vector, DataSet.ans = None,None,None,None
        DataSet.new_story, DataSet.end_story = None, None
        return DataSet

    DataSets = {}
    for story_i, story in enumerate(token_stories):
        DataSets[story_i] = []
        DataSet = init_data()
        DataSet.new_story = True
        Q_count = 0
        story_len = len(story)
        for sentc_i, sentence in enumerate(story):
            # check if this sentence is the last one, if so, set the end_story = True
            if sentc_i == story_len - 1:
                DataSet.end_story = True
            else:
                DataSet.end_story = False
            E = sentc2e(embedding, sentence, device=device)
            # check if the sentence is the question
            if word2int['<q>'] in sentence:
                DataSet.Q = E
                DataSet.q = sentence
                # acquire the ans
                DataSet.ans = token_answers[story_i][Q_count]
                DataSet.ans_vector = sentc2e(embedding, DataSet.ans, device=device)
                DataSets[story_i].append(DataSet)
                # reset after yield
                DataSet = init_data()
                DataSet.new_story = False
                Q_count += 1
            else:
                DataSet.E_s.append(E)
                DataSet.stories.append(sentence)
    return DataSets

def equalize_data_size(stories, max_sentc_len, min_q_num, max_session_len, device="cuda"):
    """
    :param stories: dict
    :param max_sentc_len: int
    :param min_q_num: int
    :param max_session_len: int
    :return:
    """
    # cut the redundant sessions
    for key, story in stories.items():
        if len(story) > min_q_num:
            stories[key] = story[0:min_q_num]
            stories[key][-1].end_story = True # set the end story is true
    # loop the stories again
    for key, story in stories.items():
        for s_i, session in enumerate(story):
            # padding sentence length both in question and each sentence
            if len(session.Q) < max_sentc_len:
                stories[key][s_i].Q = cat_zeros(session.Q, max_sentc_len, dim=0, device=device)
            for i, sentc in enumerate(session.E_s):
                if len(sentc) < max_sentc_len:
                    stories[key][s_i].E_s[i] = cat_zeros(sentc, max_sentc_len, dim=0, device=device)
            # padding or cutting session length
            if len(session.E_s) > max_session_len:
                stories[key][s_i].E_s = session.E_s[0:max_session_len]        # cut the redundant vector
            elif len(session.E_s) < max_session_len:
                for _ in range(max_session_len - len(session.E_s)):      # padding 0 sentence vector
                    stories[key][s_i].E_s.insert(0, torch.zeros_like(session.E_s[0]))
    return stories

def cat_to_full_batch(stories, stat):
    """
    :param stories: equalized size train_set/test_set
    :param stat: dict {max_sentc_len, max_q_num, min_q_num, max_session_len: int, int, int, int}
    :return: full_batch
    """
    # init the batch nametuple and their size
    full_batch = collections.namedtuple("full_batch", ["E_s", "Q", "ans", "ans_vector", "end_story", "new_story", "q", "stories"])
    full_batch.E_s = torch.empty((stat["min_q_num"], stat["max_session_len"], len(stories), stat["max_sentc_len"], EMBED_SIZE))
    full_batch.Q = torch.empty((stat["min_q_num"], len(stories), stat["max_sentc_len"], EMBED_SIZE))
    full_batch.ans = torch.empty((stat["min_q_num"], len(stories), 1))
    full_batch.ans_vector = torch.empty((stat["min_q_num"], len(stories), 1, EMBED_SIZE))
    full_batch.end_story = torch.zeros((stat["min_q_num"], len(stories)), dtype=torch.bool)
    full_batch.new_story = torch.zeros((stat["min_q_num"], len(stories)), dtype=torch.bool)
    full_batch.q = np.empty((stat["min_q_num"], len(stories)), dtype=object)
    full_batch.stories = np.empty((stat["min_q_num"], len(stories)), dtype=object)

    for story_i, story in enumerate(stories.values()):
        for session_i, session in enumerate(story):
            for sentc_i, E in enumerate(session.E_s):
                full_batch.E_s[session_i, sentc_i, story_i :, :] = E
            full_batch.Q[session_i, story_i :, :] = session.Q
            full_batch.ans[session_i, story_i :] = session.ans
            full_batch.ans_vector[session_i, story_i, :, :] = session.ans_vector
            full_batch.end_story[session_i, story_i] = session.end_story
            full_batch.new_story[session_i, story_i] = session.new_story
            full_batch.q[session_i, story_i] = session.q
            full_batch.stories[session_i, story_i] = session.stories
    return full_batch

class DataLoader:
    def __init__(self, full_batch, batch_size, shuffle, episode_len):
        """
        :param stories: nametuple: E_s, Q, ans, ans_vector, end_story, new_story, q, stories (They cat all of data)
        :param batch_size: int
        :param shuffle: Boolean
        """
        self.full_batch = full_batch
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.episode_len = episode_len

    def _data_extract(self, index):
        """
        :param index: list
        :return batch: nametuple
        """
        batch = collections.namedtuple("batch", ["E_s", "Q", "ans", "ans_vector", "end_story", "new_story", "q", "stories"])
        batch.E_s = self.full_batch.E_s[:, :, index, :, :]
        batch.Q = self.full_batch.Q[:, index, :, :]
        batch.ans = self.full_batch.ans[:, index, :]
        batch.ans_vector = self.full_batch.ans_vector[:, index, :, :]
        batch.end_story = self.full_batch.end_story[:, index]
        batch.new_story = self.full_batch.new_story[:, index]
        batch.q = self.full_batch.q[:, index]
        batch.stories = self.full_batch.stories[:, index]
        return batch

    def __iter__(self):
        tailed_num = []
        while True:
            data_index = [i for i in range(self.episode_len)]
            if self.shuffle:
                np.random.shuffle(data_index)
            data_index = tailed_num + data_index # combine both list of tailed num and head num, after shuffle
            cut_index = len(data_index) // self.batch_size * self.batch_size
            head_num = data_index[0:cut_index]
            for i in range(0, len(head_num), self.batch_size):
                batch_index = head_num[i:i+self.batch_size]
                batch = self._data_extract(batch_index)
                yield batch
            tailed_num = data_index[cut_index:]

    def create_batches(self):
        """
        :return: batches [batch]
        """
        batches = []
        for b in self:
            batches.append(b)
            if len(batches) == self.episode_len:
                return batches

class Episode_Tracker:
    def __init__(self, entNet, int2word, path, writer, episode, write_episode):
        """
        :param entNet: nn.model
        :param int2word: dict: [2: "highway", 4: "bar", 1: "poland", ... ]
        :param path: str
        :param writer: tensorboard writer
        :param episode: int, current episode
        :param write_episode: int, output txt every the k episode
        """
        self.entNet = entNet
        self.int2word = int2word
        self.path = path
        self.writer = writer
        self.write_episode = write_episode
        self.episode = episode

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def reset_story_content(self):
        # write the story with predicted ans file
        self.sentc_index = 1
        self.correct_story = True
        self.story_with_ans = ''

    def print_episode_status(self, q_count, correct, losses, type):
        episode_loss = losses / q_count
        print("{}-Episode: {}; loss: {}".format(type, self.episode, episode_loss))
        print("{}-Accuracy: {:.3f}%".format(type, correct / q_count * 100))

    def plot_episode_status(self, q_count, correct, losses, type):
        episode_loss = losses / q_count
        episode_accuracy = correct / q_count * 100
        self.writer.add_scalar(type + "-episode loss", episode_loss, self.episode)
        self.writer.add_scalar(type + "-episode accuracy", episode_accuracy, self.episode)


    def write(self, story, question, predicted_ans, ans, new_story, end_story, type):
        """
        :param story: [ [1,14,2, ... ], [2,42,2,1, ... ] ], Note: if the story is empty, it is empty list = []
        :param question: [1,12,34,2,15, ... ]
        :param predicted_ans: int
        :param ans: int
        :param new_story: boolean
        :param end_story: boolean
        :param type: "Train" / "Test"
        :return: Boolean
        """
        if new_story:
            self.reset_story_content()
            
        # check if story is correct, then output to different name of txt file
        if predicted_ans != ans:
            self.correct_story = False

        # build the self.story_with_ans
        for token_sentence in story:
            self.story_with_ans += str(self.sentc_index) + ' '
            for token in token_sentence:
                self.story_with_ans += self.int2word[token] + ' '
            self.story_with_ans += '\n'
            self.sentc_index += 1

        # build question
        self.story_with_ans += str(self.sentc_index) + ' '
        for token in question:
            self.story_with_ans += self.int2word[token] + ' '

        # build predicted ans
        self.story_with_ans += '\t'
        self.story_with_ans += self.int2word[predicted_ans] + ' '
        self.story_with_ans += '\t'

        # build ans
        self.story_with_ans += self.int2word[ans] + ' '
        self.story_with_ans += '\n'
        self.sentc_index += 1

        if end_story:
            self.append_result_txt(type)
        return True

    def append_result_txt(self, type):
        if self.correct_story:
            file_name = CORRECT_FILE_NAME.format(type, self.episode)
        else:
            file_name = INCORRECT_FILE_NAME.format(type, self.episode)
        main_path = self.path + '/' + file_name
        os.makedirs(os.path.dirname(main_path), exist_ok=True)
        with open(main_path, 'a') as f:
            f.write(self.story_with_ans)
            f.close()

    def weight_histogram(self):
        for name, param in self.entNet.named_parameters():
            self.writer.add_histogram("hist_" + name, param)

    def weight_image(self):
        for name, param in self.entNet.named_parameters():
            self.writer.add_image("image_" + name, param, global_step=self.episode, dataformats="HW")