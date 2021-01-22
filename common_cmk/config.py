from datetime import datetime
now = datetime.now()
DT_STRING = now.strftime("%y%m%d_%H%M%S")

# ------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------ CONTROL START ---------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------- #

VERSION = 4
DEVICE = "cuda"

# Skip gram / EntNet, the data index: qa1 - qa20
TRAIN_DATA_INDEX = 4

# control for skip-gram
SG_LEARNING_RATE = 0.00001
SG_SAVE_EPOCH = 1000
SG_PRINT_STEPS = 1000
SG_WRITE_EPOCH = 1000
SG_BATCH_SIZE = 64
SG_LOAD_NET = False

# control for EntNet
EntNET_LEARNING_RATE = 0.01
EntNet_SAVE_EPOCH = 500
EntNet_TEST_EPOCH = 100
EntNet_LOAD_NET = False

M_SLOTS = 20
PAD_MAX_LENGTH = 10

# Embedding
EMBED_FILE_FORMAT = "embedding-epoch-{}.data"   # Save Format
EMBED_FILE = EMBED_FILE_FORMAT.format(5000)     # load weight file

# EntNet
EntNET_FILE_FORMAT = "checkpoint-entNet-STORY{}-VERSION{}-Epoch-{}.data"    # Save Format
EntNET_FILE = EntNET_FILE_FORMAT.format(2, 2, 13400)                        # load weight file


# ------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------ CONTROL END ------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------- #

MAIN_PATH = "/home/chris/projects/201119_EntNet/docs/"
DATA_PATH = MAIN_PATH + "tasks_1-20_v1-2/en"
TRAIN_SET_NAME = {
    1: "qa1_single-supporting-fact_train.txt",
    2: "qa2_two-supporting-facts_train.txt",
    3: "qa3_three-supporting-facts_train.txt",
    4: "qa4_two-arg-relations_train.txt",
    5: "qa5_three-arg-relations_train.txt",
    6: "qa6_yes-no-questions_train.txt",
    7: "qa7_counting_train.txt",
    8: "qa8_lists-sets_train.txt",
    9: "qa9_simple-negation_train.txt",
    10: "qa10_indefinite-knowledge_train.txt",
    11: "qa11_basic-coreference_train.txt",
    12: "qa12_conjunction_train.txt",
    13: "qa13_compound-coreference_train.txt",
    14: "qa14_time-reasoning_train.txt",
    15: "qa15_basic-deduction_train.txt",
    16: "qa16_basic-induction_train.txt",
    17: "qa17_positional-reasoning_train.txt",
    18: "qa18_size-reasoning_train.txt",
    19: "qa19_path-finding_train.txt",
    20: "qa20_agents-motivations_train.txt"
}
TEST_SET_NAME = {
    1: "qa1_single-supporting-fact_test.txt",
    2: "qa2_two-supporting-facts_test.txt",
    3: "qa3_three-supporting-facts_test.txt",
    4: "qa4_two-arg-relations_test.txt",
    5: "qa5_three-arg-relations_test.txt",
    6: "qa6_yes-no-questions_test.txt",
    7: "qa7_counting_test.txt",
    8: "qa8_lists-sets_test.txt",
    9: "qa9_simple-negation_test.txt",
    10: "qa10_indefinite-knowledge_test.txt",
    11: "qa11_basic-coreference_test.txt",
    12: "qa12_conjunction_test.txt",
    13: "qa13_compound-coreference_test.txt",
    14: "qa14_time-reasoning_test.txt",
    15: "qa15_basic-deduction_test.txt",
    16: "qa16_basic-induction_test.txt",
    17: "qa17_positional-reasoning_test.txt",
    18: "qa18_size-reasoning_test.txt",
    19: "qa19_path-finding_test.txt",
    20: "qa20_agents-motivations_test.txt"
}

# for loading embedding params (Skip-gram / EntNet)
SAVE_EMBED_PATH = MAIN_PATH + str(VERSION) + "/embedding/" + str(TRAIN_DATA_INDEX)
TOKEN_COUNT = "token_count.txt"
ORIGINAL_TOKEN_COUNT = "original_token_count.txt"
INT2WORD = "int2word.txt"
WORD2INT = "word2int.txt"
SG_RESULTS = "results_epoch-{}.txt"

# for Saving and Loading params (EntNet)
SAVE_EntNET_PATH = MAIN_PATH + str(VERSION) + "/entNet_weights"

# output runs file for monitoring progress in tensorboard (Skip-Gram / EntNet)
SG_TENSORBOARD_SAVE_PATH = MAIN_PATH + str(VERSION) + "/runs/sg_qa-" + str(TRAIN_DATA_INDEX) + '_' + DT_STRING
EntNet_TENSORBOARD_SAVE_PATH = MAIN_PATH + str(VERSION) + "/runs/entnet_qa-" + str(TRAIN_DATA_INDEX) + '_' + DT_STRING
# tensorboard --logdir ~/projects/201119_EntNet/docs/3/runs --host localhost

# output result.txt for inspection (EntNet)
RESULT_CHECKING_PATH = MAIN_PATH + str(VERSION) + "/results"
INCORRECT_FILE_NAME = "S"+ str(TRAIN_DATA_INDEX) +"-{}-Incorrect-E{}.txt"
CORRECT_FILE_NAME = "S"+ str(TRAIN_DATA_INDEX) +"-{}-Correct-E{}.txt"

# Special label
QUESTION_MARK = ' <q> '
FULLSTOP = ' <fs> '
COMMA = ' <cma> '
SUBSAMPLE_LIST = ["is", "to", "the", "in", QUESTION_MARK.strip(), FULLSTOP.strip(), COMMA.strip()]