from util import *
from model import *


if __name__=='__main__':
    # Parse arguments
    DATASET, MODEL_LIST, IID_NON_COR, NON_IID_NON_COR = arg_parse()

    if DATASET == 'mnist':
        from mnist_config import *
    elif DATASET == 'fmnist':
        from fmnist_config import *
    elif DATASET == 'cifar10':
        from cifar10_config import *
    else:
        print('{} is wrong dataset! [mnist|fmnist|cifar10]'.format(DATASET))
        exit(1)

    # Load data
    tr_X, tr_y, te_X, te_y, pre_X, pre_y = load_data(data=DATASET, pre_train=PRE_TRAIN)

    # for UNCERT_FEDAVG in [False, True]:   # False, True
    for MODEL in MODEL_LIST:
        # Centralized Learning
        if MODEL == 'central':
            if IID_NON_COR or NON_IID_NON_COR:
                print('\n===================================')
                print('CUDA:', torch.cuda.is_available())
                print('MODEL:', MODEL)
                print('DATASET:', DATASET)
                print('EPOCHS:', CENTRAL_EPOCHS)
                print('BATCH_SIZE:', BATCH_SIZE)
                print('===================================\n')

                log_name = 'non_corrupted_'
                do_centralize_learning(tr_X, tr_y, te_X, te_y, BATCH_SIZE, CENTRAL_EPOCHS, log_name, DATASET)
                continue

            for DIST in DIST_LIST:
                if DIST == 'iid':
                    for COR_LABEL_RATIO in COR_LABEL_RATIO_LIST:
                        for COR_DATA_RATIO in COR_DATA_RATIO_LIST:
                            for COR_MODE in COR_MODE_LIST:
                                print('\n===================================')
                                print('CUDA:', torch.cuda.is_available())
                                print('MODEL:', MODEL)
                                print('DIST:', DIST)
                                print('DATASET:', DATASET)
                                print('EPOCHS:', CENTRAL_EPOCHS)
                                print('BATCH_SIZE:', BATCH_SIZE)
                                print('COR_MODE:', CORRUPTION_MODE[COR_MODE])
                                print('COR_LABEL_RATIO:', COR_LABEL_RATIO)
                                print('COR_DATA_RATIO:', COR_DATA_RATIO)
                                print('===================================\n')

                                log_name = DIST + '_'
                                log_name += str(int(COR_LOCAL_RATIO * 10)) + '_cor_local_'
                                log_name += str(int(COR_LABEL_RATIO * 100)) + '_cor_label_'
                                log_name += CORRUPTION_MODE[COR_MODE] + '_'

                                if COR_MODE == 2:
                                    tr_X_dict, tr_y_dict, te_X_dict, te_y_dict, _, _ = create_backdoor_iid_samples(
                                        tr_X, tr_y, te_X, te_y, target_label=TARGET_LABEL,
                                        cor_local_ratio=1.0,
                                        cor_label_ratio=COR_LABEL_RATIO,
                                        cor_data_ratio=COR_DATA_RATIO,
                                        num_of_sample=1,
                                        verbose=True,
                                        dataset=DATASET
                                    )
                                else:
                                    tr_X_dict, tr_y_dict, te_X_dict, te_y_dict = create_corrupted_iid_samples(
                                        tr_X, tr_y, te_X, te_y,
                                        cor_local_ratio=1.0,
                                        cor_label_ratio=COR_LABEL_RATIO,
                                        cor_data_ratio=COR_DATA_RATIO,
                                        mode=COR_MODE,
                                        num_of_sample=1,
                                        verbose=True,
                                        dataset=DATASET
                                    )

                                tr_X = tr_X_dict['x_train0']
                                tr_y = tr_y_dict['y_train0']
                                te_X = te_X_dict['x_test0']
                                te_y = te_y_dict['y_test0']

                                do_centralize_learning(tr_X, tr_y, te_X, te_y, BATCH_SIZE, CENTRAL_EPOCHS,
                                                       log_name, DATASET)
                else:
                    for COR_MINOR_LABEL_CNT in COR_MINOR_LABEL_CNT_LIST:
                        for COR_MINOR_DATA_RATIO in COR_MINOR_DATA_RATIO_LIST:
                            for COR_MODE in COR_MODE_LIST:
                                print('\n===================================')
                                print('CUDA:', torch.cuda.is_available())
                                print('MODEL:', MODEL)
                                print('DIST:', DIST)
                                print('DATASET:', DATASET)
                                print('EPOCHS:', CENTRAL_EPOCHS)
                                print('BATCH_SIZE:', BATCH_SIZE)
                                print('COR_MODE:', CORRUPTION_MODE[COR_MODE])
                                print('PDIST:', PDIST)
                                print('COR_MAJOR_DATA_RATIO:', COR_MAJOR_DATA_RATIO)
                                print('COR_MINOR_LABEL_CNT:', COR_MINOR_LABEL_CNT)
                                print('COR_MINOR_DATA_RATIO:', COR_MINOR_DATA_RATIO)
                                print('===================================\n')

                                log_name = DIST + '_'
                                log_name += str(int(COR_MINOR_LABEL_CNT)) + '_cor_minor_label_'
                                log_name += str(int(COR_MINOR_DATA_RATIO * 100)) + '_cor_minor_data_'
                                log_name += CORRUPTION_MODE[COR_MODE] + '_'

                                if COR_MODE == 2:   # backdoor attack
                                    tr_X_dict, tr_y_dict, te_X_dict, te_y_dict, _, _ = create_backdoor_non_iid_samples(
                                        tr_X, tr_y, te_X, te_y, TARGET_LABEL,
                                        cor_local_ratio=1.0,
                                        cor_minor_label_cnt=COR_MINOR_LABEL_CNT,
                                        cor_major_data_ratio=COR_MAJOR_DATA_RATIO,
                                        cor_minor_data_ratio=COR_MINOR_DATA_RATIO,
                                        pdist=PDIST,
                                        num_of_sample=1,
                                        verbose=True,
                                        dataset=DATASET
                                    )
                                else:
                                    tr_X_dict, tr_y_dict, te_X_dict, te_y_dict = create_corrupted_non_iid_samples(
                                        tr_X, tr_y, te_X, te_y,
                                        cor_local_ratio=1.0,
                                        cor_minor_label_cnt=COR_MINOR_LABEL_CNT,
                                        cor_major_data_ratio=COR_MAJOR_DATA_RATIO,
                                        cor_minor_data_ratio=COR_MINOR_DATA_RATIO,
                                        mode=COR_MODE,
                                        pdist=PDIST,
                                        num_of_sample=1,
                                        verbose=True,
                                        dataset=DATASET
                                    )

                                tr_X = tr_X_dict['x_train0']
                                tr_y = tr_y_dict['y_train0']
                                te_X = te_X_dict['x_test0']
                                te_y = te_y_dict['y_test0']

                                do_centralize_learning(tr_X, tr_y, te_X, te_y, BATCH_SIZE, CENTRAL_EPOCHS,
                                                       log_name, DATASET)
            continue

        # Federated Learning
        for DIST in DIST_LIST:
            cur_iid_cnt = 0
            cur_non_iid_cnt = 0

            total_common_cnt = len(UNCERT_FEDAVG_LIST) * len(COR_MODE_LIST)
            total_iid_cnt = total_common_cnt * len(COR_DATA_RATIO_LIST) * len(COR_LABEL_RATIO_LIST)
            total_non_iid_cnt = total_common_cnt * len(COR_MINOR_DATA_RATIO_LIST) * len(COR_MINOR_LABEL_CNT_LIST)

            # UG-FedAvg 적용여부 확인
            # default = 0 -> Original FedAvg
            for UNCERT_FEDAVG in UNCERT_FEDAVG_LIST:
                # IID Dist
                if DIST == 'iid':
                    # Non-corrupted Dist
                    if IID_NON_COR:
                        do_non_corruption(tr_X, tr_y, te_X, te_y,
                                          BATCH_SIZE, IID_ITERATION, IID_EPOCHS, NUM_OF_LOCAL, UNCERT_FEDAVG,
                                          DIST, DATASET)
                        break
                    # Corrupted Dataset
                    for COR_LABEL_RATIO in COR_LABEL_RATIO_LIST:
                        for COR_DATA_RATIO in COR_DATA_RATIO_LIST:
                            for COR_MODE in COR_MODE_LIST:
                                print('\n===================================')
                                print('CUDA:', torch.cuda.is_available())
                                print('UNCERT_FEDAVG:', FL_ALGO[UNCERT_FEDAVG])
                                print('MODEL:', MODEL)
                                print('DIST:', DIST)
                                print('DATASET:', DATASET)
                                print('NUM_OF_LOCAL:', NUM_OF_LOCAL)
                                print('COR_MODE:', CORRUPTION_MODE[COR_MODE])
                                print('COR_LOCAL_RATIO:', COR_LOCAL_RATIO)
                                print('COR_LABEL_RATIO:', COR_LABEL_RATIO)
                                print('COR_DATA_RATIO:', COR_DATA_RATIO)
                                print('===================================\n')

                                cur_iid_cnt += 1
                                if COR_MODE == 2:   # backdoor attack
                                    do_iid_backdoor(total_iid_cnt, cur_iid_cnt, tr_X, tr_y, te_X, te_y,
                                                    BATCH_SIZE, IID_ITERATION, IID_EPOCHS, NUM_OF_LOCAL, UNCERT_FEDAVG,
                                                    COR_LOCAL_RATIO, COR_LABEL_RATIO, COR_DATA_RATIO, TARGET_LABEL,
                                                    DATASET)
                                else:
                                    do_iid_corruption(total_iid_cnt, cur_iid_cnt, tr_X, tr_y, te_X, te_y,
                                                      BATCH_SIZE, IID_ITERATION, IID_EPOCHS, NUM_OF_LOCAL, UNCERT_FEDAVG,
                                                      COR_LOCAL_RATIO, COR_LABEL_RATIO, COR_DATA_RATIO, COR_MODE,
                                                      DATASET)
                # Non-IID Dist
                else:
                    # Non-corrupted Dataset
                    if NON_IID_NON_COR:
                        do_non_corruption(tr_X, tr_y, te_X, te_y,
                                          BATCH_SIZE, NON_IID_ITERATION, NON_IID_EPOCHS, NUM_OF_LOCAL, UNCERT_FEDAVG,
                                          DIST, DATASET)
                        break
                    # Corrupted Dataset
                    for COR_MINOR_LABEL_CNT in COR_MINOR_LABEL_CNT_LIST:
                        for COR_MINOR_DATA_RATIO in COR_MINOR_DATA_RATIO_LIST:
                            for COR_MODE in COR_MODE_LIST:
                                print('\n===================================')
                                print('CUDA:', torch.cuda.is_available())
                                print('UNCERT_FEDAVG:', FL_ALGO[UNCERT_FEDAVG])
                                print('MODEL:', MODEL)
                                print('DIST:', DIST)
                                print('DATASET:', DATASET)
                                print('NUM_OF_LOCAL:', NUM_OF_LOCAL)
                                print('COR_MODE:', CORRUPTION_MODE[COR_MODE])
                                print('PDIST:', PDIST)
                                print('COR_MAJOR_DATA_RATIO:', COR_MAJOR_DATA_RATIO)
                                print('COR_MINOR_LABEL_CNT:', COR_MINOR_LABEL_CNT)
                                print('COR_MINOR_DATA_RATIO:', COR_MINOR_DATA_RATIO)
                                print('===================================\n')

                                cur_non_iid_cnt += 1
                                if COR_MODE == 2:   # backdoor attack
                                    do_non_iid_backdoor(total_non_iid_cnt, cur_non_iid_cnt, tr_X, tr_y, te_X, te_y,
                                                        BATCH_SIZE, NON_IID_ITERATION, NON_IID_EPOCHS,
                                                        NUM_OF_LOCAL, UNCERT_FEDAVG,
                                                        COR_LOCAL_RATIO, COR_MINOR_LABEL_CNT, COR_MAJOR_DATA_RATIO,
                                                        COR_MINOR_DATA_RATIO, PDIST, TARGET_LABEL, DATASET)
                                else:
                                    do_non_iid_corruption(total_non_iid_cnt, cur_non_iid_cnt, tr_X, tr_y, te_X, te_y,
                                                          BATCH_SIZE, NON_IID_ITERATION, NON_IID_EPOCHS,
                                                          NUM_OF_LOCAL, UNCERT_FEDAVG,
                                                          COR_LOCAL_RATIO, COR_MINOR_LABEL_CNT, COR_MAJOR_DATA_RATIO,
                                                          COR_MINOR_DATA_RATIO, PDIST, COR_MODE, DATASET)
