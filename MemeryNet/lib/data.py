import torch
from collections import Counter
from codes.common_cmk import readFile

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

def preprocess_story(path, file_name):

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
    int2word, word2int = readFile.create_lookup_table(token_count, reverse=True)

    # label the word as int
    token_stories = [readFile.label_word_as_int_token(word_story, word2int) for word_story in word_stories]
    token_answers = readFile.label_word_as_int_token(answers, word2int)

    # separate the facts and questions
    # token_facts, token_questions = [], []
    # for token_story in token_stories:
    #     token_fact, token_question = readFile.target_detach(token_story, target_label=word2int['<q>'])
    #     token_facts.append(token_fact)
    #     token_questions.append(token_question)

    return token_stories, token_answers, reasons, int2word, word2int

def generate(embedding, token_stories, token_answers, word2int, fixed_length=6, device="cuda"):
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
                ans = sentc2e(embedding, token_answers[story_i][Q_count], fixed_length=1, device=device)
                yield E_s, Q, ans, new_story
                # reset after yield
                new_story = False
                E_s.clear()
                Q_count += 1
            else:
                E_s.append(E)