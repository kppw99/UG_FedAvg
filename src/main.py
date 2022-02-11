from util import *
from model import *


## COMMON #######################################
BATCH_SIZE = 64
NUM_OF_LOCAL = 10
COR_LOCAL_RATIO = 1.0
CENTRAL_EPOCHS = 20

UNCERT_FEDAVG_LIST = [0, 1, 2]  # fl_origin=0, fl_with_pre_train=1, fl_with_uncertainty=2
COR_MODE_LIST = [0, 1]  # label_flipping=0, label_shuffling=1
MODEL_LIST = ['central', 'federate']  # central, federate
DATASET_LIST = ['iid', 'non-iid']  # iid, non-iid
#################################################

## IID ##########################################
IID_NON_COR = False
IID_EPOCHS = 10     # 10 is best param
IID_ITERATION = 10  # 10 is best param

COR_LABEL_RATIO_LIST = [0.4, 0.3, 0.2, 0.1]  # 0.4, 0.3, 0.2, 0.1
COR_DATA_RATIO_LIST = [0.6, 0.5, 0.4, 0.3]  # 0.6, 0.5, 0.4, 0.3
#################################################

## Non-IID ######################################
NON_IID_NON_COR = False
NON_IID_EPOCHS = 15     # 15 is best param
NON_IID_ITERATION = 25  # 25 is best param
PDIST = 0.9
COR_MAJOR_DATA_RATIO = 0.65

COR_MINOR_LABEL_CNT_LIST = [4]
COR_MINOR_DATA_RATIO_LIST = [0.8]
#################################################


if __name__=='__main__':
    # Load data
    tr_X, tr_y, te_X, te_y, pre_X, pre_y = load_mnist_data(pre_train=True)

    # for UNCERT_FEDAVG in [False, True]:   # False, True
    for MODEL in MODEL_LIST:
        # Centralized Learning
        if MODEL == 'central':
            print('\n===================================')
            print('CUDA:', torch.cuda.is_available())
            print('MODEL:', MODEL)
            print('EPOCHS:', CENTRAL_EPOCHS)
            print('BATCH_SIZE:', BATCH_SIZE)
            print('===================================\n')

            do_centralize_learning(tr_X, tr_y, te_X, te_y, BATCH_SIZE, CENTRAL_EPOCHS)
            continue

        # Federated Learning
        for DATASET in DATASET_LIST:
            cur_iid_cnt = 0
            cur_non_iid_cnt = 0

            total_common_cnt = len(UNCERT_FEDAVG_LIST) * len(COR_MODE_LIST)
            total_iid_cnt = total_common_cnt * len(COR_DATA_RATIO_LIST) * len(COR_LABEL_RATIO_LIST)
            total_non_iid_cnt = total_common_cnt * len(COR_MINOR_DATA_RATIO_LIST) * len(COR_MINOR_LABEL_CNT_LIST)

            for UNCERT_FEDAVG in UNCERT_FEDAVG_LIST:
                # IID Dataset
                if DATASET == 'iid':
                    # Non-corrupted Dataset
                    if IID_NON_COR:
                        do_non_corruption(tr_X, tr_y, te_X, te_y,
                                          BATCH_SIZE, IID_ITERATION, IID_EPOCHS, NUM_OF_LOCAL, UNCERT_FEDAVG,
                                          DATASET)
                    # Corrupted Dataset
                    for COR_LABEL_RATIO in COR_LABEL_RATIO_LIST:
                        for COR_DATA_RATIO in COR_DATA_RATIO_LIST:
                            for COR_MODE in COR_MODE_LIST:
                                print('\n===================================')
                                print('CUDA:', torch.cuda.is_available())
                                print('UNCERT_FEDAVG:', UNCERT_FEDAVG)
                                print('MODEL:', MODEL)
                                print('DATASET:', DATASET)
                                print('NUM_OF_LOCAL:', NUM_OF_LOCAL)
                                print('COR_MODE:', COR_MODE)
                                print('COR_LOCAL_RATIO:', COR_LOCAL_RATIO)
                                print('COR_LABEL_RATIO:', COR_LABEL_RATIO)
                                print('COR_DATA_RATIO:', COR_DATA_RATIO)
                                print('===================================\n')

                                cur_iid_cnt += 1
                                do_iid_corruption(total_iid_cnt, cur_iid_cnt, tr_X, tr_y, te_X, te_y,
                                                  BATCH_SIZE, IID_ITERATION, IID_EPOCHS, NUM_OF_LOCAL, UNCERT_FEDAVG,
                                                  COR_LOCAL_RATIO, COR_LABEL_RATIO, COR_DATA_RATIO, COR_MODE)
                # Non-IID Dataset
                else:
                    # Non-corrupted Dataset
                    if NON_IID_NON_COR:
                        do_non_corruption(tr_X, tr_y, te_X, te_y,
                                          BATCH_SIZE, NON_IID_ITERATION, NON_IID_EPOCHS, NUM_OF_LOCAL, UNCERT_FEDAVG,
                                          DATASET)
                    # Corrupted Dataset
                    for COR_MINOR_LABEL_CNT in COR_MINOR_LABEL_CNT_LIST:
                        for COR_MINOR_DATA_RATIO in COR_MINOR_DATA_RATIO_LIST:
                            for COR_MODE in COR_MODE_LIST:
                                print('\n===================================')
                                print('CUDA:', torch.cuda.is_available())
                                print('UNCERT_FEDAVG:', UNCERT_FEDAVG)
                                print('MODEL:', MODEL)
                                print('DATASET:', DATASET)
                                print('NUM_OF_LOCAL:', NUM_OF_LOCAL)
                                print('COR_MODE:', COR_MODE)
                                print('PDIST:', PDIST)
                                print('COR_MAJOR_DATA_RATIO:', COR_MAJOR_DATA_RATIO)
                                print('COR_MINOR_LABEL_CNT:', COR_MINOR_LABEL_CNT)
                                print('COR_MINOR_DATA_RATIO:', COR_MINOR_DATA_RATIO)
                                print('===================================\n')

                                cur_non_iid_cnt += 1
                                do_non_iid_corruption(total_non_iid_cnt, cur_non_iid_cnt, tr_X, tr_y, te_X, te_y,
                                                      BATCH_SIZE, NON_IID_ITERATION, NON_IID_EPOCHS,
                                                      NUM_OF_LOCAL, UNCERT_FEDAVG,
                                                      COR_LOCAL_RATIO, COR_MINOR_LABEL_CNT, COR_MAJOR_DATA_RATIO,
                                                      COR_MINOR_DATA_RATIO, PDIST, COR_MODE)
