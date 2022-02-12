import gzip
import random
import pickle
import numpy as np
import pandas as pd

from pathlib import Path
from matplotlib import pyplot
from scipy.stats import entropy
import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


def _split_and_shuffle_labels(y_data, seed):
    num_of_class = len(set(y_data.tolist()))
    y_data=pd.DataFrame(y_data, columns=['label'])
    y_data['index'] = np.arange(len(y_data))
    label_dict = dict()
    cur_idx = list()

    for i in range(num_of_class):
        var_name = 'label' + str(i)
        label_info = y_data[y_data['label'] == i]
        np.random.seed(seed)
        label_info = np.random.permutation(label_info)
        label_info = pd.DataFrame(label_info, columns=['label', 'index'])
        label_dict.update({var_name: label_info })
        cur_idx.append(0)

    return label_dict, cur_idx


def _get_iid_subsamples_indices(y_data, number_of_samples, seed):
    num_of_class = len(set(y_data.tolist()))
    label_dict, cur_idx = _split_and_shuffle_labels(y_data, seed)
    sample_dict = dict()
    dist = 1.0 / num_of_class
    for i in range(number_of_samples):
        sample_name = 'sample' + str(i)
        dumb = pd.DataFrame()
        for j in range(num_of_class):
            label_name = str('label') + str(j)
            if i == (number_of_samples - 1):
                next_idx = len(label_dict[label_name])
            else:
                next_idx = int(len(label_dict[label_name]) * dist)
                next_idx += cur_idx[j]
            temp = label_dict[label_name][cur_idx[j]:next_idx]
            dumb=pd.concat([dumb, temp], axis=0)
            cur_idx[j] = next_idx
        dumb.reset_index(drop=True, inplace=True)
        sample_dict.update({sample_name: dumb})

    return sample_dict


def _get_non_iid_subsamples_indices(y_data, number_of_samples, pdist, seed):
    num_of_class = len(set(y_data.tolist()))
    label_dict, cur_idx = _split_and_shuffle_labels(y_data, seed)
    sample_dict = dict()
    for i in range(number_of_samples):
        sample_name = 'sample' + str(i)
        dumb = pd.DataFrame()
        dist1 = pdist * (2 / 3)
        dist2 = pdist - dist1
        dist3 = (1.0 - pdist) / (num_of_class - 2)
        for j in range(num_of_class):
            label_name = str('label') + str(j)
            dist = dist1 if j == i else dist2 if (j % 5) == (i % 5) else dist3
            if i == (number_of_samples - 1):
                next_idx = len(label_dict[label_name])
            else:
                next_idx = int(len(label_dict[label_name]) * dist)
                next_idx += cur_idx[j]
            temp = label_dict[label_name][cur_idx[j]:next_idx]
            dumb = pd.concat([dumb, temp], axis=0)
            cur_idx[j] = next_idx
        dumb.reset_index(drop=True, inplace=True)
        sample_dict.update({sample_name: dumb})

    return sample_dict


def _create_subsamples(sample_dict, x_data, y_data, x_name, y_name):
    x_data_dict = dict()
    y_data_dict = dict()

    for i in range(len(sample_dict)):  ### len(sample_dict)= number of samples
        xname = x_name + str(i)
        yname = y_name + str(i)
        sample_name = "sample" + str(i)

        indices = np.sort(np.array(sample_dict[sample_name]['index']))

        x_info = x_data[indices, :]
        if torch.cuda.is_available():
            x_info = x_info.cuda()
        x_data_dict.update({xname: x_info})

        y_info = y_data[indices]
        if torch.cuda.is_available():
            y_info = y_info.cuda()
        y_data_dict.update({yname: y_info})

    return x_data_dict, y_data_dict


def _create_corrupted_subsamples(sample_dict, x_data, y_data, x_name, y_name,
                                 cor_local_ratio=1.0, cor_label_ratio=0.2, cor_data_ratio=0.5, mode=1):
    x_data_dict = dict()
    y_data_dict = dict()

    # make corrupted info
    num_of_local = len(sample_dict)
    num_of_label = len(set(y_data.tolist()))
    cor_local_idx = random.sample(range(0, num_of_local), int(num_of_local * cor_local_ratio))
    cor_label_idx = random.sample(range(0, num_of_label), int(num_of_label * cor_label_ratio))
    temp = set(y_data.tolist())
    temp.difference_update(cor_label_idx)

    print('[*] Corrupted Label')
    if mode == 1:
        temp = list(temp)
        cor_vals = random.sample(temp, int(num_of_label * cor_label_ratio))
        print(cor_label_idx, '->', cor_vals)
    else:
        print(cor_label_idx, '-> random value')
    print('')

    for i in range(len(sample_dict)):  ### len(sample_dict)= number of samples
        xname = x_name + str(i)
        yname = y_name + str(i)
        sample_name = "sample" + str(i)

        indices = np.sort(np.array(sample_dict[sample_name]['index']))

        x_info = x_data[indices, :]
        if torch.cuda.is_available():
            x_info = x_info.cuda()
        x_data_dict.update({xname: x_info})

        y_info = y_data[indices]

        if i in cor_local_idx:
            val_cnt = 0
            for j in cor_label_idx:
                temp_dices = np.where(y_info == j)[0]
                cor_data_len = int(len(temp_dices) * cor_data_ratio)
                corrupted_idx = random.sample(list(temp_dices), cor_data_len)

                if mode == 1:
                    y_info[corrupted_idx] = cor_vals[val_cnt]
                    val_cnt = val_cnt + 1
                else:
                    for i in corrupted_idx:
                        temp_x = temp
                        ori_val = y_info[i].item()
                        temp_x.difference_update([ori_val])
                        y_info[i] = random.sample(temp_x, 1)[0]

        if torch.cuda.is_available():
            y_info = y_info.cuda()
        y_data_dict.update({yname: y_info})

    return x_data_dict, y_data_dict


def _create_backdoor_subsamples(sample_dict, x_data, y_data, x_name, y_name,
                                cor_label_idx, target_label, cor_local_ratio=1.0, cor_data_ratio=0.5):
    x_data_dict = dict()
    y_data_dict = dict()

    # make corrupted info
    num_of_local = len(sample_dict)
    cor_local_idx = random.sample(range(0, num_of_local), int(num_of_local * cor_local_ratio))

    # num_of_label = len(set(y_data.tolist()))
    # while(True):
    #     cor_label_idx = random.sample(range(0, num_of_label), int(num_of_label * cor_label_ratio))
    #     if not target_label in cor_label_idx:
    #         break
    # temp = set(y_data.tolist())
    # temp.difference_update(cor_label_idx)
    #
    # print('[*] Corrupted Label')
    # print(cor_label_idx, '->', target_label)
    # print('')

    # len(sample_dict) is a number of client
    for i in range(len(sample_dict)):
        xname = x_name + str(i)
        yname = y_name + str(i)
        sample_name = "sample" + str(i)

        indices = np.sort(np.array(sample_dict[sample_name]['index']))

        x_info = x_data[indices, :]
        y_info = y_data[indices]

        if i in cor_local_idx:
            for j in cor_label_idx:
                temp_dices = np.where(y_info == j)[0]
                cor_data_len = int(len(temp_dices) * cor_data_ratio)
                corrupted_idx = random.sample(list(temp_dices), cor_data_len)

                y_info[corrupted_idx] = target_label
                for idx in corrupted_idx:
                    size = 5
                    start_idx = 1
                    temp_x = x_info[idx].reshape(28, 28)
                    # trigger pattern (plus)
                    for ii in range(start_idx, start_idx+size):
                        temp_x[ii][(start_idx+size)//2] = 1.0   # vertical line
                        temp_x[(start_idx+size)//2][ii] = 1.0   # horizontal line
                    x_info[idx] = temp_x.reshape(1, 28, 28)
                    # plt.imshow(x_info[idx].reshape(28, 28))
                    # plt.show()
        if torch.cuda.is_available():
            x_info = x_info.cuda()
            y_info = y_info.cuda()
        x_data_dict.update({xname: x_info})
        y_data_dict.update({yname: y_info})

    return x_data_dict, y_data_dict


def _create_corrupted_subsamples2(sample_dict, x_data, y_data, x_name, y_name,
                                  cor_local_ratio=1.0, cor_minor_label_cnt=4,
                                  cor_major_data_ratio=0.2,
                                  cor_minor_data_ratio=0.5,
                                  mode=1):
    x_data_dict = dict()
    y_data_dict = dict()

    # make corrupted info
    num_of_local = len(sample_dict)
    num_of_label = len(set(y_data.tolist()))
    cor_local_idx = random.sample(range(0, num_of_local), int(num_of_local * cor_local_ratio))

    for i in range(len(sample_dict)):  ### len(sample_dict)= number of samples
        xname = x_name + str(i)
        yname = y_name + str(i)
        sample_name = "sample" + str(i)

        indices = np.sort(np.array(sample_dict[sample_name]['index']))

        x_info = x_data[indices, :]
        if torch.cuda.is_available():
            x_info = x_info.cuda()
        x_data_dict.update({xname: x_info})

        y_info = y_data[indices]

        if i in cor_local_idx:
            cor_major_label_idx = list()
            cor_major_label_idx.append(i)
            cor_major_label_idx.append((i + 5) % num_of_label)

            for j in cor_major_label_idx:
                temp_dices = np.where(y_info == j)[0]
                cor_data_len = int(len(temp_dices) * cor_major_data_ratio)
                corrupted_idx = random.sample(list(temp_dices), cor_data_len)

                ori_val = y_info[corrupted_idx][0]
                y_info[corrupted_idx] = (ori_val + 5) % num_of_label

            temp = set(y_data.tolist())
            temp.difference_update(cor_major_label_idx)
            cor_minor_label_idx = random.sample(temp, cor_minor_label_cnt)
            temp.difference_update(cor_minor_label_idx)
            cor_minor_vals = random.sample(temp, cor_minor_label_cnt)
            print(cor_major_label_idx, '|', cor_minor_label_idx, '->', cor_minor_vals)

            val_cnt = 0
            for j in cor_minor_label_idx:
                temp_dices = np.where(y_info == j)[0]
                cor_data_len = int(len(temp_dices) * cor_minor_data_ratio)
                corrupted_idx = random.sample(list(temp_dices), cor_data_len)

                if mode == 1:
                    y_info[corrupted_idx] = cor_minor_vals[val_cnt]
                    val_cnt = val_cnt + 1
                else:
                    cor_minor_vals = list()
                    for i in corrupted_idx:
                        temp_x = temp
                        ori_val = y_info[i].item()
                        temp_x.difference_update([ori_val])
                        y_info[i] = random.sample(temp_x, 1)[0]

        if torch.cuda.is_available():
            y_info = y_info.cuda()
        y_data_dict.update({yname: y_info})

    return x_data_dict, y_data_dict


def _print_dict(x_train_dict, y_train_dict, x_test_dict, y_test_dict, x_val_dict=None, y_val_dict=None):
    sum = 0
    print('[*] Train Dataset (x, y)')
    for idx, (x_key, y_key) in enumerate(zip(x_train_dict, y_train_dict)):
        sum += len(x_train_dict[x_key])
        print('- sample{}: {}, {}'.format(idx, len(x_train_dict[x_key]), len(y_train_dict[y_key])))
        print(': ', end='')
        for i in range(10):
            print(y_train_dict[y_key].tolist().count(i), end=' ')
        print('')
    print('# total:', sum, end='\n\n')

    sum = 0
    print('[*] Test Dataset (x, y)')
    for idx, (x_key, y_key) in enumerate(zip(x_test_dict, y_test_dict)):
        sum += len(x_test_dict[x_key])
        print('- sample{}: {}, {}'.format(idx, len(x_test_dict[x_key]), len(y_test_dict[y_key])))
        print(': ', end='')
        for i in range(10):
            print(y_test_dict[y_key].tolist().count(i), end=' ')
        print('')
    print('# total:', sum, end='\n\n')

    if x_val_dict is not None:
        sum = 0
        print('[*] Valid Dataset (x, y)')
        for idx, (x_key, y_key) in enumerate(zip(x_val_dict, y_val_dict)):
            sum += len(x_val_dict[x_key])
            print('- sample{}: {}, {}'.format(idx, len(x_val_dict[x_key]), len(y_val_dict[y_key])))
            print(': ', end='')
            for i in range(10):
                print(y_val_dict[y_key].tolist().count(i), end=' ')
            print('')
        print('# total:', sum, end='\n\n')


def load_mnist_data(path='../data/mnist.pkl.gz', seed=1, torch_tensor=True, pre_train=False):
    data_path = Path(path)
    with gzip.open(data_path, "rb") as f:
        ((x_train, y_train), (x_test, y_test)) = pickle.load(f)

    if pre_train:
        pre_rate = 0.05
        train_size = len(x_train)
        pre_data_size = int(train_size * pre_rate)

        np.random.seed(seed)
        shuffled_indices = np.random.permutation(train_size)

        pre_indices = shuffled_indices[:pre_data_size]
        tr_indices = shuffled_indices[pre_data_size:]

        x_pre_train = x_train[pre_indices]
        y_pre_train = y_train[pre_indices]

        x_train = x_train[tr_indices]
        y_train = y_train[tr_indices]

        if torch_tensor:
            x_train, y_train, x_test, y_test, x_pre_train, y_pre_train =\
                map(torch.tensor, (x_train, y_train, x_test, y_test, x_pre_train, y_pre_train))
        print(x_train.shape, y_train.shape,
              x_test.shape, y_test.shape,
              x_pre_train.shape, y_pre_train.shape)
        return x_train, y_train, x_test, y_test, x_pre_train, y_pre_train
    else:
        if torch_tensor:
            x_train, y_train, x_test, y_test = map(torch.tensor, (x_train, y_train, x_test, y_test))
        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
        return x_train, y_train, x_test, y_test, None, None


def create_non_iid_samples(x_train, y_train, x_test, y_test,
                           num_of_sample=10, pdist=0.6, seed=1, verbose=True):
    sample_dict_train = _get_non_iid_subsamples_indices(y_train, num_of_sample, pdist, seed)
    x_train_dict, y_train_dict = _create_subsamples(sample_dict_train, x_train, y_train,
                                                    'x_train', 'y_train')

    sample_dict_test = _get_non_iid_subsamples_indices(y_test, num_of_sample, pdist, seed)
    x_test_dict, y_test_dict = _create_subsamples(sample_dict_test, x_test, y_test,
                                                  'x_test', 'y_test')

    if verbose:
        _print_dict(x_train_dict, y_train_dict, x_test_dict, y_test_dict)

    return x_train_dict, y_train_dict, x_test_dict, y_test_dict


def create_corrupted_non_iid_samples(x_train, y_train, x_test, y_test,
                                     cor_local_ratio=1.0,
                                     cor_minor_label_cnt=4,
                                     cor_major_data_ratio=0.2,
                                     cor_minor_data_ratio=0.5, mode=1,
                                     num_of_sample=10, pdist=0.6, seed=1, verbose=True):
    sample_dict_train = _get_non_iid_subsamples_indices(y_train, num_of_sample, pdist, seed)
    x_train_dict, y_train_dict = _create_corrupted_subsamples2(sample_dict_train, x_train, y_train,
                                                               'x_train', 'y_train',
                                                               cor_local_ratio, cor_minor_label_cnt,
                                                               cor_major_data_ratio,
                                                               cor_minor_data_ratio, mode)

    sample_dict_test = _get_non_iid_subsamples_indices(y_test, num_of_sample, pdist, seed)
    x_test_dict, y_test_dict = _create_subsamples(sample_dict_test, x_test, y_test, 'x_test', 'y_test')

    if verbose:
        _print_dict(x_train_dict, y_train_dict, x_test_dict, y_test_dict)

    return x_train_dict, y_train_dict, x_test_dict, y_test_dict


def create_iid_samples(x_train, y_train, x_test, y_test, num_of_sample=10, seed=1, verbose=True):
    sample_dict_train = _get_iid_subsamples_indices(y_train, num_of_sample, seed)
    x_train_dict, y_train_dict = _create_subsamples(sample_dict_train, x_train, y_train,
                                                    'x_train', 'y_train')

    sample_dict_test = _get_iid_subsamples_indices(y_test, num_of_sample, seed)
    x_test_dict, y_test_dict = _create_subsamples(sample_dict_test, x_test, y_test,
                                                  'x_test', 'y_test')

    if verbose:
        _print_dict(x_train_dict, y_train_dict, x_test_dict, y_test_dict)

    return x_train_dict, y_train_dict, x_test_dict, y_test_dict


def create_corrupted_iid_samples(x_train, y_train, x_test, y_test,
                                 cor_local_ratio=1.0, cor_label_ratio=0.2, cor_data_ratio=0.5, mode=1,
                                 num_of_sample=10, seed=1, verbose=True):
    sample_dict_train = _get_iid_subsamples_indices(y_train, num_of_sample, seed)
    x_train_dict, y_train_dict = _create_corrupted_subsamples(sample_dict_train, x_train, y_train,
                                                              'x_train', 'y_train',
                                                              cor_local_ratio, cor_label_ratio,
                                                              cor_data_ratio, mode)

    sample_dict_test = _get_iid_subsamples_indices(y_test, num_of_sample, seed)
    x_test_dict, y_test_dict = _create_subsamples(sample_dict_test, x_test, y_test,
                                                  'x_test', 'y_test')

    if verbose:
        _print_dict(x_train_dict, y_train_dict, x_test_dict, y_test_dict)

    return x_train_dict, y_train_dict, x_test_dict, y_test_dict


def create_backdoor_iid_samples(x_train, y_train, x_test, y_test,
                                 cor_local_ratio=1.0, cor_label_ratio=0.2, cor_data_ratio=0.5, target_label=1,
                                 num_of_sample=10, seed=1, verbose=True):
    sample_dict_train = _get_iid_subsamples_indices(y_train, num_of_sample, seed)

    num_of_label = len(set(y_train.tolist()))
    while(True):
        cor_label_idx = random.sample(range(0, num_of_label), int(num_of_label * cor_label_ratio))
        if not target_label in cor_label_idx:
            break
    temp = set(y_train.tolist())
    temp.difference_update(cor_label_idx)

    print('[*] Corrupted Label')
    print(cor_label_idx, '->', target_label)
    print('')

    x_train_dict, y_train_dict = _create_backdoor_subsamples(sample_dict_train, x_train, y_train, 'x_train', 'y_train',
                                                             cor_label_idx, target_label,
                                                             cor_local_ratio, cor_data_ratio)

    sample_dict_test = _get_iid_subsamples_indices(y_test, num_of_sample, seed)
    x_test_dict, y_test_dict = _create_subsamples(sample_dict_test, x_test, y_test,
                                                  'x_test', 'y_test')

    x_val_dict, y_val_dict = _create_backdoor_subsamples(sample_dict_test, x_test, y_test, 'x_val', 'y_val',
                                                         cor_label_idx, target_label,
                                                         cor_local_ratio, cor_data_ratio)

    if verbose:
        _print_dict(x_train_dict, y_train_dict, x_test_dict, y_test_dict, x_val_dict, y_val_dict)

    return x_train_dict, y_train_dict, x_test_dict, y_test_dict, x_val_dict, y_val_dict


def create_dataloader(x_train, y_train, x_test, y_test, batch_size):
    train_data = None
    test_data = None

    if x_train != None and y_train != None:
        train_data = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    if x_test != None and y_test != None:
        test_data = DataLoader(TensorDataset(x_test, y_test), batch_size=1)

    return train_data, test_data


def cal_entropy(data):
    return entropy(data, base=len(data))


if __name__=='__main__':
    tr_X, tr_y, te_X, te_y = load_mnist_data()

    tr_X_iid_dict, tr_y_iid_dict, te_X_iid_dict, te_y_iid_dict = create_corrupted_iid_samples(
        tr_X, tr_y, te_X, te_y,
        cor_local_ratio=1.0, cor_label_ratio=0.2, cor_data_ratio=0.5, mode=2,
        num_of_sample=10, seed=1, verbose=True
    )

    tr_X_iid_dict, tr_y_iid_dict, te_X_iid_dict, te_y_iid_dict = create_corrupted_non_iid_samples(
        tr_X, tr_y, te_X, te_y,
        cor_local_ratio=1.0,
        cor_minor_label_cnt=1,
        cor_major_data_ratio=0.2,
        cor_minor_data_ratio=0.5, mode=1,
        num_of_sample=10, seed=1, verbose=True
    )