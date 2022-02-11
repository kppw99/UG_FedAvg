from util import *
from model import *
from config import *


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
                                if COR_MODE == 2:   # backdoor attack
                                    do_iid_backdoor(total_iid_cnt, cur_iid_cnt, tr_X, tr_y, te_X, te_y,
                                                    BATCH_SIZE, IID_ITERATION, IID_EPOCHS, NUM_OF_LOCAL, UNCERT_FEDAVG,
                                                    COR_LOCAL_RATIO, COR_LABEL_RATIO, COR_DATA_RATIO, TARGET_LABEL)
                                else:
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
