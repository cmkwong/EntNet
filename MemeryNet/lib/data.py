import torch
import torch.nn as nn
from codes.common_cmk.config import *
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

def load_file_from_SkipGram(path, embedding_file, int2word_file, word2int_file):

    SkipGram_Net = collections.namedtuple("SkipGram_Net", ["weights", "embedding", "embedding_arr", "token_count", "int2word", "word2int"])

    # Load the embedding
    print("Loading Embedding ...", end=' ')
    with open(os.path.join(path, embedding_file), "rb") as f:
        checkpoint = torch.load(f)
    weights = checkpoint['state_dict']['in_embed.weight']
    SkipGram_Net.weights = funcs.unitVector_2d(weights, dim=1)                              # normalize into unit tensor
    SkipGram_Net.embedding = nn.Embedding.from_pretrained(SkipGram_Net.weights)             # pretrained embedding network
    SkipGram_Net.embedding_arr = SkipGram_Net.embedding.weight.data.cpu().detach().numpy()  # embedding array
    print("Successful!")

    # load the int2word and word2int files
    print("Loading word2int and int2word files ...",end=' ')
    SkipGram_Net.int2word = readFile.read_dict(path, int2word_file)
    SkipGram_Net.word2int = readFile.read_dict(path, word2int_file)
    print("Successful!")

    return SkipGram_Net

def translate_story_into_token(path, file_name, word2int):

    Data = collections.namedtuple("Train_Data", ["token_stories", "token_answers", "reasons"])

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

def generate(embedding, token_stories, token_answers, word2int, fixed_length=10, device="cuda"):
    """
    :param token_stories: [ [ [1,3,5,7,8,4,9,10,19],[1,3,5,7,8,4,9], [12,3,5,7,8,14,11], ... ], ... ]
    :param token_answers: [ [12,34], ... ]
    :param word2int: dict
    :param fixed_length: int
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
            E = sentc2e(embedding, sentence, fixed_length=fixed_length, device=device)
            if word2int['<q>'] in sentence:
                GenSet.Q = E
                GenSet.q = sentence
                # acquire the ans
                GenSet.ans = token_answers[story_i][Q_count]
                GenSet.ans_vector = sentc2e(embedding, GenSet.ans, fixed_length=1, device=device)
                yield GenSet
                # reset after yield
                GenSet.new_story = False
                GenSet.E_s.clear()
                Q_count += 1
            else:
                GenSet.E_s.append(E)
                GenSet.stories.append(sentence)

def generate_data(embedding, token_stories, token_answers, word2int, fixed_length=10, device="cuda"):
    """
    :param token_stories: [ [ [1,3,5,7,8,4,9,10,19],[1,3,5,7,8,4,9], [12,3,5,7,8,14,11], ... ], ... ]
    :param token_answers: [ [12,34], ... ]
    :param word2int: dict
    :param fixed_length: int
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
            E = sentc2e(embedding, sentence, fixed_length=fixed_length, device=device)
            # check if the sentence is the question
            if word2int['<q>'] in sentence:
                DataSet.Q = E
                DataSet.q = sentence
                # acquire the ans
                DataSet.ans = token_answers[story_i][Q_count]
                DataSet.ans_vector = sentc2e(embedding, DataSet.ans, fixed_length=1, device=device)
                DataSets[story_i].append(DataSet)
                # reset after yield
                DataSet = init_data()
                DataSet.new_story = False
                Q_count += 1
            else:
                DataSet.E_s.append(E)
                DataSet.stories.append(sentence)
    return DataSets

class Episode_Tracker:
    def __init__(self, int2word, path, writer, episode, write_episode):
        """
        :param int2word: dict: [2: "highway", 4: "bar", 1: "poland", ... ]
        :param path: str
        :param writer: tensorboard writer
        :param episode: int, current episode
        :param write_episode: int, output txt every the k episode
        """
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
        with open(main_path, 'a') as f:
            f.write(self.story_with_ans)
            f.close()