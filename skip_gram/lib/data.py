from collections import Counter
import numpy as np
import random
from codes.common_cmk import readFile


def filter_out_word_from_sentcs(word_sentcs, token_count, rare_word_threshold=5):
    """
    :param sentcs: [["this", "is", "the", "east", "<fs>"], ["where", "is", "east", "<q>"], ... ]
    :param token_count: {"this": 12, "east": 23,  ...}
    :return: [["this", "is", "the", "east", "<fs>"], ["where", "is", "east", "<q>"], ... ]
    """
    # filter word each by each sentc
    filter_token_sentcs = []
    for token_sentc in word_sentcs:
        filter_token_sentc = [token for token in token_sentc if token_count[token] > rare_word_threshold]
        filter_token_sentcs.append(filter_token_sentc)
    return filter_token_sentcs

def sub_sampling_from_sentcs(word_sentcs, token_count, sampling_threshold=1e-3):
    """
    :param sentcs: [["this", "is", "the", "east", "<FS>"], ["where", "is", "east", "<Q>"], ... ]
    :param token_count: {"this": 12, "east": 23,  ...}
    :return: [["this", "is", "the", "east", "<FS>"], ["where", "is", "east", "<Q>"], ... ]
    """
    total_count = sum(token_count.values())

    freqs = {word: count / total_count for word, count in token_count.items()}
    p_drops = {word: (1 - np.sqrt(sampling_threshold / freqs[word]**2)) for word in token_count.keys()}

    sampled_sentcs = []
    for sentc in word_sentcs:
        sampled_sentc = [word for word in sentc if random.random() < (1 - p_drops[word])]
        sampled_sentcs.append(sampled_sentc)
    return sampled_sentcs

def sub_sampling_from_sentcs_special(word_sentcs, token_count):
    """
    :param sentcs: [["this", "is", "the", "east", "<FS>"], ["where", "is", "east", "<Q>"], ... ]
    :param token_count: {"this": 12, "east": 23,  ...}
    :return: [["this", "is", "the", "east", "<FS>"], ["where", "is", "east", "<Q>"], ... ]
    """
    p_drops = {word: 0.0 for word in token_count.keys()}
    for word in ["is", "to", "the"]:
        p_drops[word] = 0.9

    sampled_sentcs = []
    for sentc in word_sentcs:
        sampled_sentc = [word for word in sentc if random.random() < (1 - p_drops[word])]
        sampled_sentcs.append(sampled_sentc)
    return sampled_sentcs

def word_to_int(words, word2int):
    """
    :param words: sampled [word]
    :use: self.word2int
    :return: [index]
    """
    return [word2int[word] for word in words]

def preprocess_story(path, file_name, unknown_rate=0.01, unknown_label="<ukn>"):

    # read file
    raw_stories, answers, reasons = readFile.read_story(path, file_name)

    # label special charactor
    stories = []
    for story in raw_stories:
        stories.append([readFile.label_special_token(sentc) for sentc in story])

    # tokenize into word from the sentence
    word_stories = [readFile.tokenize_sentences(story) for story in stories]

    # get the first time token count
    token_count = Counter()
    for word_story in word_stories:
        tc = readFile.get_token_count_from_sentences(word_story)
        token_count = token_count + tc

    # create word2int and int2word
    int2word, word2int = readFile.create_lookup_table(token_count, reverse=True)

    # filter out rare words
    word_stories = [filter_out_word_from_sentcs(word_story, token_count, rare_word_threshold=1) for word_story in word_stories]

    # get the second time token count
    token_count = Counter()
    for word_story in word_stories:
        tc = readFile.get_token_count_from_sentences(word_story)
        token_count = token_count + tc

    # sample out the frequently word in stories
    word_stories = [sub_sampling_from_sentcs_special(word_story, token_count) for word_story in word_stories]
    # get the third time token count
    token_count = Counter()
    for word_story in word_stories:
        tc = readFile.get_token_count_from_sentences(word_story)
        token_count = token_count + tc

    # label the word as int
    token_stories = [readFile.label_word_as_int_token(word_story, word2int) for word_story in word_stories]

    # add the <ukn> in token_count, int2word and word2int for unknown word
    unknown_count = int(sum(token_count.values()) * unknown_rate)
    token_count[unknown_label] = unknown_count
    int2word[len(token_count)-1] = unknown_label
    word2int[unknown_label] = len(token_count)-1

    return token_stories, answers, reasons, token_count, int2word, word2int, word_stories

def get_noise_dist(token_count, reversed=True):
    """
    :param token_count: {"this": 12, "east": 23,  ...}
    :return:
    """
    total_count = sum(token_count.values())
    freqs = {word: count / total_count for word, count in token_count.items()}
    word_freqs = np.array(sorted(freqs.values(), reverse=reversed))
    unigram_dist = word_freqs / word_freqs.sum()
    noise_dist = unigram_dist ** (0.75) / np.sum(unigram_dist ** (0.75))
    return noise_dist

def get_target(words, idx, window_size):
    """
    :param words: sampled [word] / [index]
    :param idx: int
    :param window_size: int
    :return: [target_words]
    """
    R = np.random.randint(1, window_size + 1)
    start = idx - R if (idx - R) > 0 else 0
    stop = idx + R
    target_words = words[start:idx] + words[idx + 1:stop + 1]

    return list(target_words)

def get_batches(words, batch_size, window_size=3):
    """
    :param words: combined all words in one list = [11, 20, 30, 24, 10, 22, 32, 64, 22, 1, 23, ... ]
    :param batch_size: int
    :param window_size: int
    :return: yield train_index, target_index
    """
    n_batches = len(words) // batch_size

    # only full batches
    words = words[:n_batches * batch_size]

    for idx in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[idx:idx + batch_size]
        for ii in range(len(batch)):
            batch_x = batch[ii]
            batch_y = get_target(batch, ii, window_size)
            y.extend(batch_y)
            x.extend([batch_x] * len(batch_y))
        yield x, y