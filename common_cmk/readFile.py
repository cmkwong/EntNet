import re
from collections import Counter
import numpy as np
import random

def read_txt_generator(path, file_name):
    full_path = path + '/' + file_name
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

def label_special_token(sentence):
    """
    :param sentence: "Mary journeyed to the hallway."
    :return:         "Mary journeyed to the hallway <FS> "
    """
    sentence = re.sub('\.', " <fs> ", sentence)
    sentence = re.sub('\?', " <q> ", sentence)
    return sentence

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

def label_word_as_int_token(word_sentcs, word2int):
    """
    :param word_sentcs: [["this", "is", "the", "east", "<FS>"], ["where", "is", "east", "<Q>"], ... ] (It is one story)
    :param word2int: {"this": 12, "east": 23,  ...}
    :return: [[11, 20, 30, 24, 10], [34, 20, 24, 28], ... ]
    """
    token_sentcs = []
    for sentc in word_sentcs:
        token_sentc = [word2int[word] for word in sentc]
        token_sentcs.append(token_sentc)
    return token_sentcs

def target_detach(int_tokens, target_label):
    """
    :param int_tokens: [ [1,3,5,7,8,4,9,10,19],[1,3,5,7,8,4,9], [12,3,5,7,8,14,11] ...]
    :param target_label: String / Number such as '<q>', pop out which the target label is included
    :return: int_tokens = [ [1,3,5,7,8,4,9,10,19],[1,3,5,7,8,4,9], ... ], target = [ [12,3,5,7,8,14,11], ... ]
    """
    target = []
    int_tokens_copy = int_tokens.copy()
    pop_count = 0

    for pos, sentc in enumerate(int_tokens_copy):
        pos = pos - pop_count
        if target_label in sentc:
            target.append(int_tokens.pop(pos))
            pop_count += 1
    return int_tokens, target
