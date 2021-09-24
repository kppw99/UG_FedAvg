from util import *
from model import *


def do_FL(_dataset, _iteration, _epochs, _batch_size,
          _tr_X_dict, _tr_y_dict, _te_X_dict, _te_y_dict, _te_X, _te_y,
          _num_of_local, _log_name, cur_cnt, total_cnt, verbose=False):
    print('\n===================================')
    print('MODEL: Federated Learning')
    print('DATASET: {} ({}/{})'.format(_dataset, cur_cnt, total_cnt))
    print('ITERATION:', _iteration)
    print('EPOCHS:', _epochs)
    print('BATCH_SIZE:', _batch_size)
    print('LOG_NAME:', log_name)
    print('===================================\n')

    main_model, local_models = federated_learning(
        _tr_X_dict, _tr_y_dict, _te_X_dict, _te_y_dict, _te_X, _te_y,
        _num_of_local,
        iteration=_iteration,
        epochs=_epochs,
        batch_size=_batch_size,
        log_name=_log_name,
        verbose=verbose
    )

    create_eval_report(main_model, _te_X, _te_y)
    compare_local_and_merged_model(main_model, local_models,
                                   _te_X_dict, _te_y_dict)
    model_name = '../data/model/' + log_name + '_main_model'
    model_name += time.strftime("_%Y%m%d-%H%M%S")
    save_model(main_model, model_name)


if __name__=='__main__':
    BATCH_SIZE = 32
    NUM_OF_LOCAL = 10
    COR_LOCAL_RATIO = 1.0
    CENTRAL_EPOCHS = 20

    COR_MODE_LIST = [1, 2]  # 1, 2
    MODEL_LIST = ['federated']   # central, federate
    DATASET_LIST = ['non-iid']   # iid, non-iid

    ## IID ##########################################
    IID_EPOCHS = 10
    IID_ITERATION = 20

    COR_LABEL_RATIO_LIST = [0.4, 0.3, 0.2]
    COR_DATA_RATIO_LIST = [0.6, 0.5, 0.4, 0.3]
    #################################################

    ## Non-IID ######################################
    NON_IID_EPOCHS = 15
    NON_IID_ITERATION = 25
    PDIST = 0.9
    COR_MAJOR_DATA_RATIO = 0.4

    COR_MINOR_LABEL_CNT_LIST = [4, 3, 2]
    COR_MINOR_DATA_RATIO_LIST = [0.6, 0.5, 0.4, 0.3]
    #################################################

    # Load data
    tr_X, tr_y, te_X, te_y, pre_X, pre_y = load_mnist_data(pre_train=True)

    for MODEL in MODEL_LIST:
        # Centralized Learning
        if MODEL == 'central':
            print('\n===================================')
            print('MODEL:', MODEL)
            print('EPOCHS:', CENTRAL_EPOCHS)
            print('BATCH_SIZE:', BATCH_SIZE)
            print('===================================\n')

            if torch.cuda.is_available():
                tr_X = tr_X.cuda()
                tr_y = tr_y.cuda()
                te_X = te_X.cuda()
                te_y = te_y.cuda()

            centralized_model = centralized_learning(
                tr_X, tr_y, te_X, te_y,
                epochs=CENTRAL_EPOCHS,
                batch_size=BATCH_SIZE
            )

            centralized_report = create_eval_report(centralized_model, te_X, te_y)
            model_name = '../data/model/centralized_' + str(CENTRAL_EPOCHS) + '_model'
            model_name += time.strftime("_%Y%m%d-%H%M%S")
            save_model(centralized_model, model_name)

        # Federated Learning
        for DATASET in DATASET_LIST:
            # IID Dataset
            if DATASET == 'iid':
                # Non-corrupted
                tr_X_dict, tr_y_dict, te_X_dict, te_y_dict = create_corrupted_iid_samples(
                    tr_X, tr_y, te_X, te_y,
                    cor_local_ratio=0.0,
                    num_of_sample=NUM_OF_LOCAL,
                    verbose=True
                )

                log_name = 'federated_' + DATASET + '_non_corrupted'

                do_FL(DATASET, IID_ITERATION, IID_EPOCHS, BATCH_SIZE,
                      tr_X_dict, tr_y_dict, te_X_dict, te_y_dict,
                      te_X, te_y, NUM_OF_LOCAL, log_name,
                      1, 1,
                      verbose=False)

                # Corrupted Dataset
                cur_cnt = 0
                total_cnt = len(COR_LABEL_RATIO_LIST) * len(COR_DATA_RATIO_LIST) * len(COR_MODE_LIST)
                for COR_LABEL_RATIO in COR_LABEL_RATIO_LIST:
                    for COR_DATA_RATIO in COR_DATA_RATIO_LIST:
                        for COR_MODE in COR_MODE_LIST:
                            print('\n===================================')
                            print('MODEL:', MODEL)
                            print('DATASET:', DATASET)
                            print('NUM_OF_LOCAL:', NUM_OF_LOCAL)
                            print('COR_MODE:', COR_MODE)
                            print('COR_LOCAL_RATIO:', COR_LOCAL_RATIO)
                            print('COR_LABEL_RATIO:', COR_LABEL_RATIO)
                            print('COR_DATA_RATIO:', COR_DATA_RATIO)
                            print('===================================\n')

                            tr_X_dict, tr_y_dict, te_X_dict, te_y_dict = create_corrupted_iid_samples(
                                tr_X, tr_y, te_X, te_y,
                                cor_local_ratio=COR_LOCAL_RATIO,
                                cor_label_ratio=COR_LABEL_RATIO,
                                cor_data_ratio=COR_DATA_RATIO,
                                mode=COR_MODE,
                                num_of_sample=NUM_OF_LOCAL,
                                verbose=True
                            )

                            log_name = 'federated_' + DATASET + '_'
                            log_name += str(int(COR_LABEL_RATIO * 10)) + '_'
                            log_name += str(int(COR_DATA_RATIO * 100)) + '_'
                            log_name += str(int(COR_MODE))

                            cur_cnt += 1

                            do_FL(DATASET, IID_ITERATION, IID_EPOCHS, BATCH_SIZE,
                                  tr_X_dict, tr_y_dict, te_X_dict, te_y_dict,
                                  te_X, te_y, NUM_OF_LOCAL, log_name,
                                  cur_cnt, total_cnt,
                                  verbose=False)
            # Non-IID Dataset
            else:
                # Non-corrupted
                tr_X_dict, tr_y_dict, te_X_dict, te_y_dict = create_corrupted_non_iid_samples(
                    tr_X, tr_y, te_X, te_y,
                    cor_local_ratio=0.0,
                    num_of_sample=NUM_OF_LOCAL,
                    verbose=True
                )

                log_name = 'federated_' + DATASET + '_non_corrupted'

                do_FL(DATASET, NON_IID_ITERATION, NON_IID_EPOCHS, BATCH_SIZE,
                      tr_X_dict, tr_y_dict, te_X_dict, te_y_dict,
                      te_X, te_y, NUM_OF_LOCAL, log_name,
                      1, 1,
                      verbose=False)

                # Corrupted Dataset
                cur_cnt = 0
                total_cnt = len(COR_MINOR_LABEL_CNT_LIST) * len(COR_MINOR_DATA_RATIO_LIST) * len(COR_MODE_LIST)
                for COR_MINOR_LABEL_CNT in COR_MINOR_LABEL_CNT_LIST:
                    for COR_MINOR_DATA_RATIO in COR_MINOR_DATA_RATIO_LIST:
                        for COR_MODE in COR_MODE_LIST:
                            print('\n===================================')
                            print('MODEL:', MODEL)
                            print('DATASET:', DATASET)
                            print('NUM_OF_LOCAL:', NUM_OF_LOCAL)
                            print('COR_MODE:', COR_MODE)
                            print('PDIST:', PDIST)
                            print('COR_MAJOR_DATA_RATIO:', COR_MAJOR_DATA_RATIO)
                            print('COR_MINOR_LABEL_CNT:', COR_MINOR_LABEL_CNT)
                            print('COR_MINOR_DATA_RATIO:', COR_MINOR_DATA_RATIO)
                            print('===================================\n')

                            tr_X_dict, tr_y_dict, te_X_dict, te_y_dict = create_corrupted_non_iid_samples(
                                tr_X, tr_y, te_X, te_y,
                                cor_local_ratio=COR_LOCAL_RATIO,
                                cor_minor_label_cnt=COR_MINOR_LABEL_CNT,
                                cor_major_data_ratio=COR_MAJOR_DATA_RATIO,
                                cor_minor_data_ratio=COR_MINOR_DATA_RATIO,
                                mode=COR_MODE,
                                pdist=PDIST,
                                num_of_sample=NUM_OF_LOCAL,
                                verbose=True
                            )

                            log_name = 'federated_' + DATASET + '_'
                            log_name += str(int(COR_MINOR_LABEL_CNT)) + '_'
                            log_name += str(int(COR_MINOR_DATA_RATIO * 100)) + '_'
                            log_name += str(int(COR_MODE))

                            cur_cnt += 1

                            do_FL(DATASET, NON_IID_ITERATION, NON_IID_EPOCHS, BATCH_SIZE,
                                  tr_X_dict, tr_y_dict, te_X_dict, te_y_dict,
                                  te_X, te_y, NUM_OF_LOCAL, log_name,
                                  cur_cnt, total_cnt,
                                  verbose=False)
