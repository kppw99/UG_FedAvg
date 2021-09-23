import time

from util import *

from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from torch import nn
import torch.nn.functional as F


use_cuda = torch.cuda.is_available()


class CNN4FL(nn.Module):
    def __init__(self):
        # 항상 torch.nn.Module을 상속받고 시작
        super(CNN4FL, self).__init__()
        conv1 = nn.Conv2d(1, 6, 5, 1)  # 6@24*24
        # activation ReLU
        pool1 = nn.MaxPool2d(2)  # 6@12*12
        conv2 = nn.Conv2d(6, 16, 5, 1)  # 16@8*8
        # activation ReLU
        pool2 = nn.MaxPool2d(2)  # 16@4*4

        self.conv_module = nn.Sequential(
            conv1,
            nn.ReLU(),
            pool1,
            conv2,
            nn.ReLU(),
            pool2
        )

        fc1 = nn.Linear(16 * 4 * 4, 120)
        # activation ReLU
        fc2 = nn.Linear(120, 84)
        # activation ReLU
        fc3 = nn.Linear(84, 10)

        self.fc_module = nn.Sequential(
            fc1,
            nn.ReLU(),
            fc2,
            nn.ReLU(),
            fc3
        )

        # gpu로 할당
        if use_cuda:
            self.conv_module = self.conv_module.cuda()
            self.fc_module = self.fc_module.cuda()

    def forward(self, x):
        out = self.conv_module(x)  # @16*4*4
        # make linear
        dim = 1
        for d in out.size()[1:]:  # 16, 4, 4
            dim = dim * d
        out = out.view(-1, dim)
        out = self.fc_module(out)
        return F.softmax(out, dim=1)

def _train(model, train_loader, criterion, optimizer):
    model.train()
    train_loss = 0.0
    correct = 0

    for data, target in train_loader:
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        prediction = output.argmax(dim=1, keepdim=True)
        correct += prediction.eq(target.view_as(prediction)).sum().item()

    return train_loss / len(train_loader), correct / len(train_loader.dataset)


def _evaluate(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)

            test_loss += criterion(output, target).item()
            prediction = output.argmax(dim=1, keepdim=True)
            correct += prediction.eq(target.view_as(prediction)).sum().item()

    test_loss /= len(test_loader)
    correct /= len(test_loader.dataset)

    return (test_loss, correct)


def _sync_model(main_model, model_dict):
    cnt = len(model_dict)
    name_of_models = list(model_dict.keys())

    with torch.no_grad():
        for i in range(cnt):
            model_dict[name_of_models[i]].load_state_dict(main_model.state_dict())

    return model_dict


def _train_local_model(model_dict, criterion_dict, optimizer_dict,
                       x_train_dict, y_train_dict, x_test_dict, y_test_dict,
                       number_of_samples, epochs, batch_size, verbose=True):
    name_of_x_train_sets = list(x_train_dict.keys())
    name_of_y_train_sets = list(y_train_dict.keys())
    name_of_x_test_sets = list(x_test_dict.keys())
    name_of_y_test_sets = list(y_test_dict.keys())

    name_of_models = list(model_dict.keys())
    name_of_optimizers = list(optimizer_dict.keys())
    name_of_criterions = list(criterion_dict.keys())

    logs = list()
    if verbose is False:
        for i in tqdm(range(number_of_samples), desc='Train local models'):
            train_data = DataLoader(TensorDataset(x_train_dict[name_of_x_train_sets[i]],
                                                  y_train_dict[name_of_y_train_sets[i]]),
                                    batch_size=batch_size, shuffle=True)

            test_data = DataLoader(TensorDataset(x_test_dict[name_of_x_test_sets[i]],
                                                 y_test_dict[name_of_y_test_sets[i]]), batch_size=1)

            model = model_dict[name_of_models[i]]
            criterion = criterion_dict[name_of_criterions[i]]
            optimizer = optimizer_dict[name_of_optimizers[i]]

            epoch_logs = list()
            for epoch in range(epochs):
                train_loss, train_accuracy = _train(model, train_data, criterion, optimizer)
                test_loss, test_accuracy = _evaluate(model, test_data, criterion)
                epoch_logs.append([train_loss, train_accuracy, test_loss, test_accuracy])
            logs.append(epoch_logs)
    else:
        for i in range(number_of_samples):
            train_data = DataLoader(TensorDataset(x_train_dict[name_of_x_train_sets[i]],
                                                  y_train_dict[name_of_y_train_sets[i]]),
                                    batch_size=batch_size, shuffle=True)

            test_data = DataLoader(TensorDataset(x_test_dict[name_of_x_test_sets[i]],
                                                 y_test_dict[name_of_y_test_sets[i]]), batch_size=1)

            model = model_dict[name_of_models[i]]
            criterion = criterion_dict[name_of_criterions[i]]
            optimizer = optimizer_dict[name_of_optimizers[i]]

            print('Local_{}'.format(i))
            print('--------------------------------------------')
            epoch_logs = list()
            for epoch in range(epochs):
                train_loss, train_accuracy = _train(model, train_data, criterion, optimizer)
                test_loss, test_accuracy = _evaluate(model, test_data, criterion)
                print("[epoch {}/{}]".format(epoch + 1, epochs)
                      + " train_loss: {:0.4f}, train_acc: {:0.4f}".format(train_loss, train_accuracy)
                      + " | test_loss: {:0.4f}, test_acc: {:0.4f}".format(test_loss, test_accuracy))
                epoch_logs.append([train_loss, train_accuracy, test_loss, test_accuracy])
            logs.append(epoch_logs)
            print('--------------------------------------------\n')

    return logs


def _update_main_model(main_model, model_dict):
    node_states = list()
    node_cnt = len(model_dict)
    name_of_models = list(model_dict.keys())

    with torch.no_grad():
        main_state = main_model.state_dict()

        for key in main_state:
            total_state = 0.0
            for i in range(node_cnt):
                total_state += model_dict[name_of_models[i]].state_dict()[key]
            main_state[key] = total_state / float(node_cnt)

    main_model.load_state_dict(main_state)

    return main_model


def _create_local_models(number_of_samples=10, lr=0.01, momentum=0.9):
    model_dict = dict()
    optimizer_dict = dict()
    criterion_dict = dict()

    for i in range(number_of_samples):
        model_name = 'model' + str(i)
        model_info = CNN4FL()
        model_dict.update({model_name: model_info})

        optimizer_name = 'optimizer' + str(i)
        optimizer_info = torch.optim.SGD(model_info.parameters(), lr=lr, momentum=momentum)
        optimizer_dict.update({optimizer_name: optimizer_info})

        criterion_name = 'criterion' + str(i)
        criterion_info = nn.CrossEntropyLoss()
        criterion_dict.update({criterion_name: criterion_info})

    return model_dict, optimizer_dict, criterion_dict


def federated_learning(x_train_dict, y_train_dict, x_test_dict, y_test_dict, x_test, y_test,
                       number_of_samples, iteration, epochs, batch_size, log_name, verbose=False):
    main_model = CNN4FL()
    main_criterion = nn.CrossEntropyLoss()

    local_model_dict, local_optimizer_dict, local_criterion_dict = _create_local_models(number_of_samples)
    _, test_data = create_dataloader(None, None, x_test, y_test, batch_size)

    main_logs = list()
    local_logs = list()
    for i in range(iteration):
        print('[*] Iteration: {}/{}'.format(str(i + 1), str(iteration)))
        local_model_dict = _sync_model(main_model, local_model_dict)
        local_log = _train_local_model(local_model_dict, local_criterion_dict, local_optimizer_dict,
                                       x_train_dict, y_train_dict, x_test_dict, y_test_dict,
                                       number_of_samples, epochs=epochs, batch_size=batch_size,
                                       verbose=verbose)
        main_model = _update_main_model(main_model, local_model_dict)
        test_loss, test_accuracy = _evaluate(main_model, test_data, main_criterion)
        print("[iter {}/{}]".format(i + 1, iteration)
              + " main_loss: {:0.4f}, main_acc: {:0.4f}".format(test_loss, test_accuracy))
        create_eval_report(main_model, x_test, y_test, printable=verbose)
        main_logs.append([test_loss, test_accuracy])
        local_logs.append(local_log)

    log_dict = {
        'main': main_logs,
        'local': local_logs
    }

    filetime = time.strftime("_%Y%m%d-%H%M%S")
    temp_name = '_' + str(number_of_samples) + '_' + str(iteration) + '_' + str(epochs) + '_' + str(batch_size)
    filename = '../data/exp_result/' + log_name + temp_name + filetime + '.pkl'

    with open(filename, 'wb') as f:
        pickle.dump(log_dict, f)

    return main_model, local_model_dict


def create_eval_report(model, x_test, y_test, printable=True):
    y_pred = model(x_test)
    y_pred = y_pred.argmax(dim=1)
    report = classification_report(y_test, y_pred, digits=4)
    if printable:
        print(report)

    return report


def centralized_learning(x_train, y_train, x_test, y_test, epochs, batch_size):
    model = CNN4FL()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    train_data, test_data = create_dataloader(x_train, y_train, x_test, y_test, batch_size)
    print("------ Centralized Model ------")
    logs = list()
    for epoch in range(epochs):
        central_train_loss, central_train_accuracy = _train(model, train_data, criterion, optimizer)
        central_test_loss, central_test_accuracy = _evaluate(model, test_data, criterion)

        print("[epoch {}/{}]".format(epoch + 1, epochs)
              + " train loss: {:0.4f}".format(central_train_loss)
              + ", train accuracy: {:7.4f}".format(central_train_accuracy)
              + " | test loss: {:0.4f}".format(central_test_loss)
              + ", test accuracy: {:7.4f}".format(central_test_accuracy))
        logs.append([central_train_loss, central_train_accuracy, central_test_loss, central_test_accuracy])
    print("------ Training finished ------")

    filetime = time.strftime("_%Y%m%d-%H%M%S")
    filename = '../data/exp_result/' + 'central_' + str(epochs) + '_' + str(batch_size) + filetime + '.pkl'

    log_dict = {
        'main': logs
    }

    with open(filename, 'wb') as f:
        pickle.dump(log_dict, f)

    return model


def compare_local_and_merged_model(main_model, local_model_dict,
                                   x_test_dict, y_test_dict,
                                   printable=True):
    number_of_samples = len(local_model_dict)
    accuracy_table = pd.DataFrame(data=np.zeros((number_of_samples, 3)),
                                  columns=['local', 'local_ind_model', 'merged_main_model'])
    for i, (m, x, y) in enumerate(zip(local_model_dict, x_test_dict, y_test_dict)):
        local_model = local_model_dict[m]
        x_test = x_test_dict[x]
        y_test = y_test_dict[y]

        y_pred = local_model(x_test).argmax(dim=1)
        local_accuracy = accuracy_score(y_pred, y_test)

        y_pred = main_model(x_test).argmax(dim=1)
        main_accuracy = accuracy_score(y_pred, y_test)

        accuracy_table.loc[i, 'local'] = 'local ' + str(i)
        accuracy_table.loc[i, 'local_ind_model'] = local_accuracy
        accuracy_table.loc[i, 'merged_main_model'] = main_accuracy

    if printable:
        print(accuracy_table)

    return accuracy_table


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(path):
    model = CNN4FL()
    model.load_state_dict(torch.load(path))

    return model