import re
from collections import Counter
import numpy as np
import random

def label_special_token(sentence):
    """
    :param sentence: "Mary journeyed to the hallway."
    :return:         "Mary journeyed to the hallway <FS> "
    """
    sentence = re.sub('\.', " <fs> ", sentence)
    sentence = re.sub('\?', " <q> ", sentence)
    return sentence

def read_txt_generator(path, file_name):
    txt = []
    full_path = path + '/' + file_name
    file = open
    with open(full_path, 'r') as file:
        for line in file:
            yield line
    file.close()

def story_start_end(sentence_list):
    """
    :param sentence_list: [   [ "1 Mary journeyed to the hallway.",
                                "2 Where is Sandra?\tgarden\t1",
                                "3 Where is Peter?\tOffice\t2",
                                "..."], ... ]
    :return: [(0,10), (11, 15), (16, 24), ...]
    """
    story_loc = []
    starts = []
    for index, sent in enumerate(sentence_list):
        story_num = int(sent.split(' ')[0])
        if story_num == 1:
            starts.append(index)
    for index in range(len(starts)):
        if index != len(starts) - 1:
            story_loc.append((starts[index], starts[index + 1]))
        else:
            story_loc.append((starts[index], len(sentence_list) + 1))

    return story_loc

def read_story(path, file_name):
    """
    :param path: String
    :param file_name: String
    :return: storys =   [   ["Mary journeyed to the hallway.",
                            "Where is Sandra?",
                            "Where is Peter?",
                            "..."], ... ]
             answer  =  [   ["kitchen", "office"...], ...]
             reasons =  [   [7 8], [5 6], ...]
    """
    stories, answers, reasons = [], [], []
    story, answer, reason = [], [], []
    sentences = []

    for line in read_txt_generator(path, file_name):
        line = line.replace('\n', '')
        sentences.append(line)

    story_loc = story_start_end(sentences)

    for start, end in story_loc:

        sentence_lot = sentences[start:end]

        for sentence in sentence_lot:
            sentence = sentence.split(' ', maxsplit=1)[1]
            if '\t' in sentence:
                parts = sentence.split('\t')
                story.append(parts[0])
                answer.append(parts[1])
                reason.append(parts[2])
            else:
                story.append(sentence)

        stories.append(story)
        answers.append(answer)
        reasons.append(reason)
        # reset the buffer
        story, answer, reason = [], [], []

    return stories, answers, reasons

def tokenize_sentences(story):
    """
    :param story:                   [   "Mary journeyed to the hallway <fs> ",
                                        "Where is Sandra <fs> ",
                                        "Where is Peter <fs> ",
                                        ...]
    :return: word_sentences =       [   ["mary", "journeyed", "to", "the", "hallway" "<fs>"],
                                        ["where", "is", "sandra", "<fs>" ],
                                        ["where", "is", "peter", <fs>"],
                                        "..."], ... ]
    """
    word_sentences = []
    for sentc in story:
        token_sentence = [word.lower() for word in sentc.split(' ') if word]
        word_sentences.append(token_sentence)
    return word_sentences

def get_token_count_from_sentences(word_sentences):
    """
    :param: word_sentences =       [   ["mary", "journeyed", "to", "the", "hallway" "<fs>"],
                                        ["where", "is", "sandra", "<fs>" ],
                                        ["where", "is", "peter", <fs>"],
                                        "..."], ... ]
    :return: token_count = {"theater": 12, ...}
    """
    words = []
    for sentc in word_sentences:
        sub_sentcs = [word for word in sentc]
        words.extend(sub_sentcs)
    token_count = Counter(words)
    return token_count

def create_lookup_table(token_count, reverse=True):
    """
    :param token_count: token_count
    :return: int2word, word2int
    :NOTE: the int is ascending sorted by the word frequency
    """
    sorted_words = sorted(token_count, key=token_count.get, reverse=reverse)
    int2word = {ii: word for ii, word in enumerate(sorted_words)}
    word2int = {word: ii for ii, word in int2word.items()}
    return int2word, word2int

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

def label_word_as_token(word_sentcs, word2int):
    """
    :param word_sentcs: [["this", "is", "the", "east", "<FS>"], ["where", "is", "east", "<Q>"], ... ]
    :param word2int: {"this": 12, "east": 23,  ...}
    :return: [[11, 20, 30, 24, 10], [34, 20, 24, 28], ... ]
    """
    token_sentcs = []
    for sentc in word_sentcs:
        token_sentc = [word2int[word] for word in sentc]
        token_sentcs.append(token_sentc)
    return token_sentcs

def word_to_int(words, word2int):
    """
    :param words: sampled [word]
    :use: self.word2int
    :return: [index]
    """
    return [word2int[word] for word in words]

def preprocess_story(path, file_name):

    # read file
    raw_stories, answers, reasons = read_story(path, file_name)

    # label special charactor
    stories = []
    for story in raw_stories:
        stories.append([label_special_token(sentc) for sentc in story])

    # tokenize into word from the sentence
    word_stories = [tokenize_sentences(story) for story in stories]

    # get the first time token count
    token_count = Counter()
    for word_story in word_stories:
        tc = get_token_count_from_sentences(word_story)
        token_count = token_count + tc

    # create word2int and int2word
    int2word, word2int = create_lookup_table(token_count, reverse=True)

    # filter out rare words
    word_stories = [filter_out_word_from_sentcs(word_story, token_count, rare_word_threshold=1) for word_story in word_stories]

    # get the second time token count
    token_count = Counter()
    for word_story in word_stories:
        tc = get_token_count_from_sentences(word_story)
        token_count = token_count + tc

    # sample out the frequently word in stories
    word_stories = [sub_sampling_from_sentcs_special(word_story, token_count) for word_story in word_stories]
    # get the third time token count
    token_count = Counter()
    for word_story in word_stories:
        tc = get_token_count_from_sentences(word_story)
        token_count = token_count + tc

    # label the word as int
    token_stories = [label_word_as_token(word_story, word2int) for word_story in word_stories]

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

