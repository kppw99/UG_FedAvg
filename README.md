# Uncertainty_Guided_FedAvg
This project is underway. We will update the README file after finishing this project.

### Prerequisite (for configure)
- mnist_config.py
- fmnist_config.py
- cifar10_config.py
```
## COMMON #######################################
BATCH_SIZE = 64
NUM_OF_LOCAL = 10
COR_LOCAL_RATIO = 1.0
CENTRAL_EPOCHS = 50

PRE_TRAIN = True
UNCERT_FEDAVG_LIST = [0, 1, 2]  # fl_origin=0, fl_with_pre_train=1, fl_with_uncertainty=2
COR_MODE_LIST = [0, 1, 2]  # label_flipping=0, label_shuffling=1, backdoor=2
DIST_LIST = ['iid', 'non-iid']  # iid, non-iid
TARGET_LABEL = 1    # for backdoor attack
#################################################

## IID ##########################################
IID_EPOCHS = 10     # 10 is best param
IID_ITERATION = 10  # 10 is best param

COR_LABEL_RATIO_LIST = [0.3]  # 0.4, 0.3, 0.2, 0.1
COR_DATA_RATIO_LIST = [0.2]   # 0.6, 0.5, 0.4, 0.3
#################################################

## Non-IID ######################################
# NON_IID_NON_COR = None # True or False
NON_IID_EPOCHS = 15         # 15 is best param
NON_IID_ITERATION = 25      # 25 is best param
PDIST = 0.9
COR_MAJOR_DATA_RATIO = 0.65 # 0.65 is best param

COR_MINOR_LABEL_CNT_LIST = [4]      # 4
COR_MINOR_DATA_RATIO_LIST = [0.8]   # 0.8
#################################################
```

### Usage
```
python3 ./main.py --dataset [mnist|fmnist|cifar10] --model [central, federate] --corrupt [True|False]
```
- dataset: dataset [mnist|fmnist|cifar10] (default: mnist)
- model: model list [central, federate] (default: [central, federate])
- corrupt: data corruption [True|False] (default: True)

## About
This program is authored and maintained by **Sanghoon(Kevin) Jeon** and **Kyung Ho Park**.
> Email: kppw99@gmail.com, kp@socar.kr

> GitHub[@UG_FedAvg](https://github.com/kppw99/UG_FedAvg)
