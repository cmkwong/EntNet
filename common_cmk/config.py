from datetime import datetime
now = datetime.now()
DT_STRING = now.strftime("%y%m%d%H%M%S")

# ------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------ CONTROL START ---------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------- #

VERSION = 12
DEVICE = "cuda"

# Skip gram / EntNet, the data index: qa1 - qa20
TRAIN_DATA_INDEX = 2

# control for skip-gram
SG_LEARNING_RATE = 0.00001
EMBED_SIZE = 64
SG_SAVE_EPOCH = 1000
SG_PRINT_STEPS = 1000
SG_WRITE_EPOCH = 1000
SG_BATCH_SIZE = 64
SG_LOAD_NET = False

# control for EntNet
EntNET_LEARNING_RATE = 0.001
EntNet_SAVE_EPOCH = 500
EntNet_TEST_EPOCH = 100
EntNet_RESULT_EPOCH = 1000
EntNet_STATE_EPOCH = 10000
WEIGHT_HIST_EPOCH = 100
WEIGHT_IMAGE_EPOCH = 100
EntNet_LOAD_NET = False
EntNet_LOAD_INIT = False
PAD_MAX_LENGTH = 7

# Embedding (EntNet for loading)
TIME = "210301101946"
EMBED_FILE_FORMAT = "T{}-embedding-Epoch{}-Dim{}.data"   # Save Format
EMBED_FILE = EMBED_FILE_FORMAT.format(TIME, 0, 64)     # load weight file (Excep this for loaded by Skip-gram)
LOADED_INT2WORD = "T" + TIME + "-int2word.txt"  # loaded for entNet
LOADED_WORD2INT = "T" + TIME + "-word2int.txt"  # loaded for entNet

# EntNet (Weight and init weight)
EntNET_FILE_FORMAT = "T{}_checkpoint-entNet-STORY{}-VERSION{}-Epoch{}.data"    # Save Format
EntNET_FILE = EntNET_FILE_FORMAT.format("210301103241", 2, 11, 5500)           # load weight file
EntNET_INIT_FILE_FORMAT = "T{}_init-entNet-VERSION{}.data"
EntNET_INIT_FILE = EntNET_INIT_FILE_FORMAT.format("210228084838", 11) # 7 padding


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
TOKEN_COUNT = "T" + DT_STRING + "-token_count.txt"
ORIGINAL_TOKEN_COUNT = "T" + DT_STRING + "-original_token_count.txt"
SAVED_INT2WORD = "T" + DT_STRING + "-int2word.txt"
SAVED_WORD2INT = "T" + DT_STRING + "-word2int.txt"
SG_RESULTS = "T" + DT_STRING + "-results_Epoch{}.txt"

# for Saving and Loading params (EntNet)
SAVE_EntNET_PATH = MAIN_PATH + str(VERSION) + "/entNet_weights"

# EntNET params checkpoint file save name
EntNET_INIT_FILE_SAVED = EntNET_INIT_FILE_FORMAT.format(DT_STRING, VERSION)
EntNET_FILE_SAVED = EntNET_FILE_FORMAT.format(DT_STRING, TRAIN_DATA_INDEX, VERSION, "{}")

# Skip-Gram params file save name
EMBED_FILE_SAVED = EMBED_FILE_FORMAT.format(DT_STRING, "{}", EMBED_SIZE)

# output runs file for monitoring progress in tensorboard (Skip-Gram / EntNet)
SG_TENSORBOARD_SAVE_PATH = MAIN_PATH + str(VERSION) + "/runs/T" + DT_STRING + "_sg_qa" + str(TRAIN_DATA_INDEX) + '_Dim' + str(EMBED_SIZE) + '_lr' + str(SG_LEARNING_RATE) + '_V' + str(VERSION)
EntNet_TENSORBOARD_SAVE_PATH = MAIN_PATH + str(VERSION) + "/runs/T" + DT_STRING + "_entnet_qa" + str(TRAIN_DATA_INDEX) + '_lr' +  str(EntNET_LEARNING_RATE) + '_V' + str(VERSION)
# tensorboard --logdir ~/projects/201119_EntNet/docs/3/runs --host localhost

# output result.txt for inspection (EntNet)
RESULT_CHECKING_PATH = MAIN_PATH + str(VERSION) + "/results/T" + DT_STRING
INCORRECT_FILE_NAME = "QA"+ str(TRAIN_DATA_INDEX) +"-{}-Incorrect-E{}.txt"
CORRECT_FILE_NAME = "QA"+ str(TRAIN_DATA_INDEX) +"-{}-Correct-E{}.txt"

# output state, params and gradient for inspection (EntNet)
STATE_CHECKING_PATH = MAIN_PATH + str(VERSION) + "/state"
STATE_MATRICS = DT_STRING + "_state_e{}.pkl"
PARAMS_MATRICS = DT_STRING + "_params_e{}.pkl"
GRADIEND_MATRICS =  DT_STRING + "_gradient_e{}.pkl"
STATE_PATH = DT_STRING + "_state-path_e{}.pkl"

# Special label
QUESTION_MARK = ' <q> '
FULLSTOP = ' <fs> '
COMMA = ' <cma> '
SUBSAMPLE_LIST = ["is", "to", "the", "in", QUESTION_MARK.strip(), FULLSTOP.strip(), COMMA.strip()]