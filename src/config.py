## COMMON #######################################
BATCH_SIZE = 128
NUM_OF_LOCAL = 10
COR_LOCAL_RATIO = 1.0
CENTRAL_EPOCHS = 200

DATASET = 'cifar10'     # mnist, cifar10
PRE_TRAIN = False
UNCERT_FEDAVG_LIST = [0]  # fl_origin=0, fl_with_pre_train=1, fl_with_uncertainty=2
COR_MODE_LIST = [0, 1, 2]  # label_flipping=0, label_shuffling=1, backdoor=2
MODEL_LIST = ['central']  # central, federate
DIST_LIST = ['iid', 'non-iid']  # iid, non-iid
TARGET_LABEL = 1
#################################################

## IID ##########################################
IID_NON_COR = False # True or False
IID_EPOCHS = 2     # 10 is best param
IID_ITERATION = 2  # 10 is best param

COR_LABEL_RATIO_LIST = [0.3]  # 0.4, 0.3, 0.2, 0.1
COR_DATA_RATIO_LIST = [0.2]  # 0.6, 0.5, 0.4, 0.3
#################################################

## Non-IID ######################################
NON_IID_NON_COR = False # True or False
NON_IID_EPOCHS = 2     # 15 is best param
NON_IID_ITERATION = 2  # 25 is best param
PDIST = 0.9
COR_MAJOR_DATA_RATIO = 0.6 # 0.65 is best param

COR_MINOR_LABEL_CNT_LIST = [4]      # 4
COR_MINOR_DATA_RATIO_LIST = [0.8]   # 0.8
#################################################