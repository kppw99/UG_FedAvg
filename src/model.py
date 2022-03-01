import time
import os
from util import *
# from resnet import *
# from vgg import *
import torchvision.models as models

from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from torch import nn
import torch.nn.functional as F


FL_ALGO = ['origin', 'with_pretrain', 'uncertainty']
CORRUPTION_MODE = ['label_flipping', 'label_shuffling', 'backdoor']
use_cuda = torch.cuda.is_available()


class CNN4FL_MNIST(nn.Module):
    def __init__(self):
        # 항상 torch.nn.Module을 상속받고 시작
        super(CNN4FL_MNIST, self).__init__()
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


class Lenet5(torch.nn.Module):
    def __init__(self):
        super(Lenet5, self).__init__()

        self.l1 = torch.nn.Conv2d(1, 6, kernel_size=5, padding=0, stride=1)
        self.x1 = torch.nn.Tanh()
        self.l2 = torch.nn.AvgPool2d(kernel_size=2, padding=0, stride=2)

        self.l3 = torch.nn.Conv2d(6, 16, kernel_size=5, padding=0, stride=1)
        self.x2 = torch.nn.Tanh()
        self.l4 = torch.nn.AvgPool2d(kernel_size=2, padding=0, stride=2)

        self.l5 = torch.nn.Flatten()

        self.l6 = torch.nn.Linear(16 * 5 * 5, 120, bias=True)
        self.x3 = torch.nn.Tanh()

        self.l7 = torch.nn.Linear(120, 84, bias=True)
        self.x4 = torch.nn.Tanh()

        self.l8 = torch.nn.Linear(84, 10, bias=True)

    def forward(self, x):
        out = self.l1(x)
        out = self.x1(out)
        out = self.l2(out)
        out = self.l3(out)
        out = self.x2(out)
        out = self.l4(out)

        out = out.view(out.size(0), -1)

        out = self.l6(out)
        out = self.x3(out)
        out = self.l7(out)
        out = self.x4(out)
        out = self.l8(out)
        return out


def _train(model, train_loader, criterion, optimizer):
    model.train()
    model = torch.nn.DataParallel(model)
    train_loss = 0.0
    correct = 0

    if use_cuda:
        model.cuda()
        criterion = criterion.cuda()

    for data, target in train_loader:
        if use_cuda:
            data = data.float().cuda()
            target = target.cuda()

        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        prediction = output.argmax(dim=1, keepdim=True)
        correct += prediction.eq(target.view_as(prediction)).sum().item()

    return train_loss / len(train_loader), correct / len(train_loader.dataset)


def temp_train(model, train_loader, criterion, optimizer):
    model.train()
    model = torch.nn.DataParallel(model)
    train_loss = 0.0
    correct = 0

    if use_cuda:
        model.cuda()
        criterion = criterion.cuda()

    for data, target in train_loader:
        if use_cuda:
            data = data.float().cuda()
            target = target.cuda()

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

    if use_cuda:
        model.cuda()
        criterion = criterion.cuda()

    with torch.no_grad():
        for data, target in test_loader:
            if use_cuda:
                data = data.float().cuda()
                target = target.cuda()
            output = model(data)

            test_loss += criterion(output, target).item()
            prediction = output.argmax(dim=1, keepdim=True)
            correct += prediction.eq(target.view_as(prediction)).sum().item()

    test_loss /= len(test_loader)
    correct /= len(test_loader.dataset)

    return (test_loss, correct)


def temp_evaluate(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0

    if use_cuda:
        model.cuda()
        criterion = criterion.cuda()

    with torch.no_grad():
        for data, target in test_loader:
            if use_cuda:
                data = data.cuda()
                target = target.cuda()

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
                       number_of_samples, epochs, batch_size, verbose=True, dataset='mnist'):
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
            if dataset=='mnist':
                train_data = DataLoader(TensorDataset(x_train_dict[name_of_x_train_sets[i]],
                                                      y_train_dict[name_of_y_train_sets[i]]),
                                        batch_size=batch_size, shuffle=True)

                test_data = DataLoader(TensorDataset(x_test_dict[name_of_x_test_sets[i]],
                                                     y_test_dict[name_of_y_test_sets[i]]), batch_size=1)
            elif dataset=='fmnist':
                workers = 4
                transform = transforms.Compose([transforms.ToPILImage(),
                                                transforms.Resize((35, 35)),
                                                transforms.ToTensor()])

                train_dataset = CustomTensorDataset(tensors=(x_train_dict[name_of_x_train_sets[i]],
                                                             y_train_dict[name_of_y_train_sets[i]]),
                                                    transform=transform)
                train_data = torch.utils.data.DataLoader(train_dataset,
                                                         batch_size=batch_size, shuffle=True,
                                                         num_workers=workers, pin_memory=True)

                test_dataset = CustomTensorDataset(tensors=(x_test_dict[name_of_x_test_sets[i]],
                                                            y_test_dict[name_of_y_test_sets[i]]),
                                                   transform=transform)
                test_data = torch.utils.data.DataLoader(test_dataset,
                                                        batch_size=batch_size, shuffle=False,
                                                        num_workers=workers, pin_memory=True)
            elif dataset=='cifar10':
                workers = 4
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

                train_transform = transforms.Compose([transforms.ToPILImage(),
                                                      transforms.RandomHorizontalFlip(),
                                                      transforms.RandomCrop(32, 4),
                                                      transforms.ToTensor(),
                                                      normalize
                                                      ])

                test_transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                                     normalize
                                                     ])

                train_dataset = CustomTensorDataset(tensors=(x_train_dict[name_of_x_train_sets[i]],
                                                             y_train_dict[name_of_y_train_sets[i]]),
                                                    transform=train_transform)
                train_data = torch.utils.data.DataLoader(train_dataset,
                                                         batch_size=batch_size, shuffle=True,
                                                         num_workers=workers, pin_memory=True)

                test_dataset = CustomTensorDataset(tensors=(x_test_dict[name_of_x_test_sets[i]],
                                                            y_test_dict[name_of_y_test_sets[i]]),
                                                   transform=test_transform)
                test_data = torch.utils.data.DataLoader(test_dataset,
                                                        batch_size=batch_size, shuffle=False,
                                                        num_workers=workers, pin_memory=True)

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
            if dataset == 'mnist':
                train_data = DataLoader(TensorDataset(x_train_dict[name_of_x_train_sets[i]],
                                                      y_train_dict[name_of_y_train_sets[i]]),
                                        batch_size=batch_size, shuffle=True)

                test_data = DataLoader(TensorDataset(x_test_dict[name_of_x_test_sets[i]],
                                                     y_test_dict[name_of_y_test_sets[i]]), batch_size=1)
            elif dataset == 'fmnist':
                workers = 4
                transform = transforms.Compose([transforms.ToPILImage(),
                                                transforms.Resize((35, 35)),
                                                transforms.ToTensor()])

                train_dataset = CustomTensorDataset(tensors=(x_train_dict[name_of_x_train_sets[i]],
                                                             y_train_dict[name_of_y_train_sets[i]]),
                                                    transform=transform)
                train_data = torch.utils.data.DataLoader(train_dataset,
                                                         batch_size=batch_size, shuffle=True,
                                                         num_workers=workers, pin_memory=True)

                test_dataset = CustomTensorDataset(tensors=(x_test_dict[name_of_x_test_sets[i]],
                                                            y_test_dict[name_of_y_test_sets[i]]),
                                                   transform=transform)
                test_data = torch.utils.data.DataLoader(test_dataset,
                                                        batch_size=batch_size, shuffle=False,
                                                        num_workers=workers, pin_memory=True)
            elif dataset == 'cifar10':
                workers = 4
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

                train_transform = transforms.Compose([transforms.ToPILImage(),
                                                      transforms.RandomHorizontalFlip(),
                                                      transforms.RandomCrop(32, 4),
                                                      transforms.ToTensor(),
                                                      normalize
                                                      ])

                test_transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                                     normalize
                                                     ])

                train_dataset = CustomTensorDataset(tensors=(x_train_dict[name_of_x_train_sets[i]],
                                                             y_train_dict[name_of_y_train_sets[i]]),
                                                    transform=train_transform)
                train_data = torch.utils.data.DataLoader(train_dataset,
                                                         batch_size=batch_size, shuffle=True,
                                                         num_workers=workers, pin_memory=True)

                test_dataset = CustomTensorDataset(tensors=(x_test_dict[name_of_x_test_sets[i]],
                                                            y_test_dict[name_of_y_test_sets[i]]),
                                                   transform=test_transform)
                test_data = torch.utils.data.DataLoader(test_dataset,
                                                        batch_size=batch_size, shuffle=False,
                                                        num_workers=workers, pin_memory=True)

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


def _uncert_train_local_model(model_dict, criterion_dict, optimizer_dict,
                              x_train_dict, y_train_dict, x_test_dict, y_test_dict,
                              number_of_samples, epochs, batch_size,
                              uncert_threshold=0.2,
                              verbose=True, dataset='mnist'):
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
            if dataset == 'mnist':
                train_pre_data = DataLoader(TensorDataset(x_train_dict[name_of_x_train_sets[i]],
                                                          y_train_dict[name_of_y_train_sets[i]]),
                                            batch_size=1, shuffle=False)
            elif dataset == 'fmnist':
                workers = 4
                transform = transforms.Compose([transforms.ToPILImage(),
                                                transforms.Resize((35, 35)),
                                                transforms.ToTensor()])

                train_dataset = CustomTensorDataset(tensors=(x_train_dict[name_of_x_train_sets[i]],
                                                             y_train_dict[name_of_y_train_sets[i]]),
                                                    transform=transform)
                train_pre_data = torch.utils.data.DataLoader(train_dataset,
                                                         batch_size=1, shuffle=False,
                                                         num_workers=workers, pin_memory=True)
            elif dataset == 'cifar10':
                workers = 4
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

                train_transform = transforms.Compose([transforms.ToPILImage(),
                                                      transforms.RandomHorizontalFlip(),
                                                      transforms.RandomCrop(32, 4),
                                                      transforms.ToTensor(),
                                                      normalize
                                                      ])

                train_dataset = CustomTensorDataset(tensors=(x_train_dict[name_of_x_train_sets[i]],
                                                             y_train_dict[name_of_y_train_sets[i]]),
                                                    transform=train_transform)
                train_pre_data = torch.utils.data.DataLoader(train_dataset,
                                                         batch_size=1, shuffle=False,
                                                         num_workers=workers, pin_memory=True)

            pre_model = model_dict[name_of_models[i]]
            pre_model.eval()
            if use_cuda:
                pre_model.cuda()

            new_data = list()
            new_target = list()
            with torch.no_grad():
                true_cnt = 0
                false_cnt = 0
                for data, target in train_pre_data:
                    output = pre_model(data)
                    prediction = output.argmax(dim=1, keepdim=True)

                    # Add uncertainty algorithm
                    if prediction[0] == target[0]:
                        condition = True
                        true_cnt += 1
                    else:
                        entropy = cal_entropy(output[0].tolist())
                        if entropy < uncert_threshold:
                            condition = False
                            false_cnt += 1
                        else:
                            condition = True
                            true_cnt += 1

                    if condition:
                        new_data.append(data[0])
                        new_target.append(target[0])
            new_data = torch.stack(new_data)
            new_target = torch.stack(new_target)

            print(true_cnt, false_cnt)

            if dataset == 'mnist':
                train_data = DataLoader(TensorDataset(new_data, new_target),
                                        batch_size=batch_size, shuffle=True)

                test_data = DataLoader(TensorDataset(x_test_dict[name_of_x_test_sets[i]],
                                                     y_test_dict[name_of_y_test_sets[i]]),
                                       batch_size=1)
            elif dataset == 'fmnist':
                workers = 4
                transform = transforms.Compose([transforms.ToPILImage(),
                                                transforms.Resize((35, 35)),
                                                transforms.ToTensor()])

                train_dataset = CustomTensorDataset(tensors=(new_data, new_target),
                                                    transform=transform)
                train_data = torch.utils.data.DataLoader(train_dataset,
                                                         batch_size=batch_size, shuffle=True,
                                                         num_workers=workers, pin_memory=True)

                test_dataset = CustomTensorDataset(tensors=(x_test_dict[name_of_x_test_sets[i]],
                                                            y_test_dict[name_of_y_test_sets[i]]),
                                                   transform=transform)
                test_data = torch.utils.data.DataLoader(test_dataset,
                                                        batch_size=1, shuffle=False,
                                                        num_workers=workers, pin_memory=True)
            elif dataset == 'cifar10':
                workers = 4
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

                train_transform = transforms.Compose([transforms.ToPILImage(),
                                                      transforms.RandomHorizontalFlip(),
                                                      transforms.RandomCrop(32, 4),
                                                      transforms.ToTensor(),
                                                      normalize
                                                      ])

                test_transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                                     normalize
                                                     ])

                train_dataset = CustomTensorDataset(tensors=(new_data, new_target),
                                                    transform=train_transform)
                train_data = torch.utils.data.DataLoader(train_dataset,
                                                         batch_size=batch_size, shuffle=True,
                                                         num_workers=workers, pin_memory=True)

                test_dataset = CustomTensorDataset(tensors=(x_test_dict[name_of_x_test_sets[i]],
                                                            y_test_dict[name_of_y_test_sets[i]]),
                                                   transform=test_transform)
                test_data = torch.utils.data.DataLoader(test_dataset,
                                                        batch_size=1, shuffle=False,
                                                        num_workers=workers, pin_memory=True)

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
            if dataset == 'mnist':
                train_pre_data = DataLoader(TensorDataset(x_train_dict[name_of_x_train_sets[i]],
                                                          y_train_dict[name_of_y_train_sets[i]]),
                                            batch_size=1, shuffle=False)
            elif dataset == 'fmnist':
                workers = 4
                transform = transforms.Compose([transforms.ToPILImage(),
                                                transforms.Resize((35, 35)),
                                                transforms.ToTensor()])

                train_dataset = CustomTensorDataset(tensors=(x_train_dict[name_of_x_train_sets[i]],
                                                             y_train_dict[name_of_y_train_sets[i]]),
                                                    transform=transform)
                train_pre_data = torch.utils.data.DataLoader(train_dataset,
                                                             batch_size=1, shuffle=False,
                                                             num_workers=workers, pin_memory=True)
            elif dataset == 'cifar10':
                workers = 4
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

                train_transform = transforms.Compose([transforms.ToPILImage(),
                                                      transforms.RandomHorizontalFlip(),
                                                      transforms.RandomCrop(32, 4),
                                                      transforms.ToTensor(),
                                                      normalize
                                                      ])

                train_dataset = CustomTensorDataset(tensors=(x_train_dict[name_of_x_train_sets[i]],
                                                             y_train_dict[name_of_y_train_sets[i]]),
                                                    transform=train_transform)
                train_pre_data = torch.utils.data.DataLoader(train_dataset,
                                                             batch_size=1, shuffle=False,
                                                             num_workers=workers, pin_memory=True)

            pre_model = model_dict[name_of_models[i]]
            pre_model.eval()
            if use_cuda:
                pre_model.cuda()

            new_data = list()
            new_target = list()
            with torch.no_grad():
                true_cnt = 0
                false_cnt = 0
                for data, target in train_pre_data:
                    output = pre_model(data)
                    prediction = output.argmax(dim=1, keepdim=True)

                    # Add uncertainty algorithm
                    if prediction[0] == target[0]:
                        condition = True
                        true_cnt += 1
                    else:
                        entropy = cal_entropy(output[0].tolist())
                        if entropy < uncert_threshold:
                            condition = False
                            false_cnt += 1
                        else:
                            condition = True
                            true_cnt += 1

                    if condition:
                        new_data.append(data[0])
                        new_target.append(target[0])
            new_data = torch.stack(new_data)
            new_target = torch.stack(new_target)

            print(true_cnt, false_cnt)

            if dataset == 'mnist':
                train_data = DataLoader(TensorDataset(new_data, new_target),
                                        batch_size=batch_size, shuffle=True)

                test_data = DataLoader(TensorDataset(x_test_dict[name_of_x_test_sets[i]],
                                                     y_test_dict[name_of_y_test_sets[i]]),
                                       batch_size=1)
            elif dataset == 'fmnist':
                workers = 4
                transform = transforms.Compose([transforms.ToPILImage(),
                                                transforms.Resize((35, 35)),
                                                transforms.ToTensor()])

                train_dataset = CustomTensorDataset(tensors=(new_data, new_target),
                                                    transform=transform)
                train_data = torch.utils.data.DataLoader(train_dataset,
                                                         batch_size=batch_size, shuffle=True,
                                                         num_workers=workers, pin_memory=True)

                test_dataset = CustomTensorDataset(tensors=(x_test_dict[name_of_x_test_sets[i]],
                                                            y_test_dict[name_of_y_test_sets[i]]),
                                                   transform=transform)
                test_data = torch.utils.data.DataLoader(test_dataset,
                                                        batch_size=1, shuffle=False,
                                                        num_workers=workers, pin_memory=True)
            elif dataset == 'cifar10':
                workers = 4
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

                train_transform = transforms.Compose([transforms.ToPILImage(),
                                                      transforms.RandomHorizontalFlip(),
                                                      transforms.RandomCrop(32, 4),
                                                      transforms.ToTensor(),
                                                      normalize
                                                      ])

                test_transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                                     normalize
                                                     ])

                train_dataset = CustomTensorDataset(tensors=(new_data, new_target),
                                                    transform=train_transform)
                train_data = torch.utils.data.DataLoader(train_dataset,
                                                         batch_size=batch_size, shuffle=True,
                                                         num_workers=workers, pin_memory=True)

                test_dataset = CustomTensorDataset(tensors=(x_test_dict[name_of_x_test_sets[i]],
                                                            y_test_dict[name_of_y_test_sets[i]]),
                                                   transform=test_transform)
                test_data = torch.utils.data.DataLoader(test_dataset,
                                                        batch_size=1, shuffle=False,
                                                        num_workers=workers, pin_memory=True)

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


def _create_local_models(dataset='mnist', number_of_samples=10, lr=0.01, momentum=0.9):
    model_dict = dict()
    optimizer_dict = dict()
    criterion_dict = dict()

    for i in range(number_of_samples):
        model_name = 'model' + str(i)
        if dataset=='mnist':
            model_info = CNN4FL_MNIST()
        elif dataset=='fmnist':
            model_info = Lenet5()
        elif dataset=='cifar10':
            # model_info = resnet32()
            model_info = models.resnet18(pretrained=False)
            model_info.fc = nn.Linear(512, 10)
        
        model_dict.update({model_name: model_info})

        optimizer_name = 'optimizer' + str(i)
        optimizer_info = torch.optim.SGD(model_info.parameters(), lr=lr, momentum=momentum)
        optimizer_dict.update({optimizer_name: optimizer_info})

        criterion_name = 'criterion' + str(i)
        criterion_info = nn.CrossEntropyLoss()
        criterion_dict.update({criterion_name: criterion_info})

    return model_dict, optimizer_dict, criterion_dict


def federated_learning(x_train_dict, y_train_dict, x_test_dict, y_test_dict, x_test, y_test,
                       number_of_samples, iteration, epochs, batch_size, log_name,
                       dataset='mnist', pre_train=True,
                       verbose=False):
    if pre_train:
        if dataset == 'mnist':
            if use_cuda:
                main_model = load_model('../data/model_mnist/pre_train_model', dataset=dataset)
            else:
                main_model = load_model('../data/model_mnist/pre_train_model_no_cuda', dataset=dataset)
        elif dataset == 'fmnist':
            if use_cuda:
                main_model = load_model('../data/model_fmnist/pre_train_model', dataset=dataset)
            else:
                main_model = load_model('../data/model_fmnist/pre_train_model_no_cuda', dataset=dataset)
        elif dataset == 'cifar10':
            if use_cuda:
                main_model = load_model('../data/model_cifar10/pre_train_model', dataset=dataset)
            else:
                main_model = load_model('../data/model_cifar10/pre_train_model_no_cuda', dataset=dataset)
    else:
        if dataset=='mnist':
            main_model = CNN4FL_MNIST()
        elif dataset=='fmnist':
            main_model = Lenet5()
        elif dataset=='cifar10':
            # main_model = resnet32()
            main_model = models.resnet18(pretrained=False)
            main_model.fc = nn.Linear(512, 10)
            if use_cuda:
                main_model.cuda()

    main_criterion = nn.CrossEntropyLoss()

    local_model_dict, local_optimizer_dict, local_criterion_dict = _create_local_models(
        dataset=dataset, number_of_samples=number_of_samples)
    if use_cuda:
        x_test = x_test.cuda()
        y_test = y_test.cuda()
    _, test_data = create_dataloader(None, None, x_test, y_test, batch_size, dataset=dataset)

    main_logs = list()
    local_logs = list()
    for i in range(iteration):
        print('[*] Iteration: {}/{}'.format(str(i + 1), str(iteration)))
        local_model_dict = _sync_model(main_model, local_model_dict)

        local_log = _train_local_model(local_model_dict, local_criterion_dict, local_optimizer_dict,
                                       x_train_dict, y_train_dict, x_test_dict, y_test_dict,
                                       number_of_samples, epochs=epochs, batch_size=batch_size,
                                       verbose=verbose, dataset=dataset)

        main_model = _update_main_model(main_model, local_model_dict)
        test_loss, test_accuracy = _evaluate(main_model, test_data, main_criterion)
        print("[iter {}/{}]".format(i + 1, iteration)
              + " main_loss: {:0.4f}, main_acc: {:0.4f}".format(test_loss, test_accuracy))
        _, acc = create_eval_report(main_model, x_test, y_test, printable=verbose, dataset=dataset)
        main_logs.append([test_loss, test_accuracy])
        local_logs.append(local_log)

    log_dict = {
        'main': main_logs,
        'local': local_logs
    }

    filetime = time.strftime("_%Y%m%d-%H%M%S")
    temp_name = '_' + dataset + '_' + str(number_of_samples) + '_' + str(iteration) + '_' + str(epochs) + '_' + str(batch_size)
    temp_name += '_' + str(acc)
    filename = '../data/exp_result/' + log_name + temp_name + filetime + '.pkl'

    with open(filename, 'wb') as f:
        pickle.dump(log_dict, f)
    print('\n==> SAVE LOG:', filename, end='\n\n')

    return main_model, local_model_dict


def uncert_federated_learning(x_train_dict, y_train_dict, x_test_dict, y_test_dict, x_test, y_test,
                              number_of_samples, iteration, epochs, batch_size, log_name,
                              dataset='mnist', uncert_threshold=0.2,
                              verbose=False):
    if dataset=='mnist':
        if use_cuda:
            main_model = load_model('../data/model_mnist/pre_train_model', dataset=dataset)
        else:
            main_model = load_model('../data/model_mnist/pre_train_model_no_cuda', dataset=dataset)
    elif dataset=='fmnist':
        if use_cuda:
            main_model = load_model('../data/model_fmnist/pre_train_model', dataset=dataset)
        else:
            main_model = load_model('../data/model_fmnist/pre_train_model_no_cuda', dataset=dataset)
    elif dataset=='cifar10':
        if use_cuda:
            main_model = load_model('../data/model_cifar10/pre_train_model', dataset=dataset)
        else:
            main_model = load_model('../data/model_cifar10/pre_train_model_no_cuda', dataset=dataset)

    main_criterion = nn.CrossEntropyLoss()

    local_model_dict, local_optimizer_dict, local_criterion_dict = _create_local_models(
        dataset=dataset, number_of_samples=number_of_samples)
    if use_cuda:
        x_test = x_test.cuda()
        y_test = y_test.cuda()
    _, test_data = create_dataloader(None, None, x_test, y_test, batch_size, dataset=dataset)

    main_logs = list()
    local_logs = list()
    for i in range(iteration):
        print('[*] Iteration: {}/{}'.format(str(i + 1), str(iteration)))
        local_model_dict = _sync_model(main_model, local_model_dict)

        local_log = _uncert_train_local_model(local_model_dict,
                                              local_criterion_dict, local_optimizer_dict,
                                              x_train_dict, y_train_dict,
                                              x_test_dict, y_test_dict,
                                              number_of_samples,
                                              epochs=epochs,
                                              batch_size=batch_size,
                                              uncert_threshold=uncert_threshold,
                                              verbose=verbose,
                                              dataset=dataset)

        main_model = _update_main_model(main_model, local_model_dict)
        test_loss, test_accuracy = _evaluate(main_model, test_data, main_criterion)
        print("[iter {}/{}]".format(i + 1, iteration)
              + " main_loss: {:0.4f}, main_acc: {:0.4f}".format(test_loss, test_accuracy))
        _, acc = create_eval_report(main_model, x_test, y_test, printable=verbose, dataset=dataset)
        main_logs.append([test_loss, test_accuracy])
        local_logs.append(local_log)

    log_dict = {
        'main': main_logs,
        'local': local_logs
    }

    filetime = time.strftime("_%Y%m%d-%H%M%S")
    temp_name = '_' + dataset + '_' + str(number_of_samples) + '_' + str(iteration) + '_' + str(epochs) + '_' + str(batch_size)
    temp_name += '_' + str(acc)
    filename = '../data/exp_result/' + log_name + temp_name + filetime + '.pkl'

    with open(filename, 'wb') as f:
        pickle.dump(log_dict, f)
    print('\n==> SAVE LOG:', filename, end='\n\n')

    return main_model, local_model_dict


def create_eval_report(model, x_test, y_test, printable=True, dataset='mnist'):
    if dataset == 'cifar10':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            normalize
        ])
        test_dataset = CustomTensorDataset(tensors=(x_test, y_test), transform=test_transform)

        test_data = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=len(x_test), shuffle=False)

        x_test = next(iter(test_data))[0]
        y_test = next(iter(test_data))[1]
    elif dataset=='fmnist':
        transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((35, 35)), transforms.ToTensor()])
        test_dataset = CustomTensorDataset(tensors=(x_test, y_test), transform=transform)

        test_data = torch.utils.data.DataLoader(test_dataset, batch_size=len(x_test), shuffle=False)

        x_test = next(iter(test_data))[0]
        y_test = next(iter(test_data))[1]

    if use_cuda:
        x_test = x_test.float().cuda()
        y_test = y_test.cuda()
        
    with torch.no_grad():
        y_pred = model(x_test)
        y_pred = y_pred.argmax(dim=1)
    
    y_test = y_test.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    
    report = classification_report(y_test, y_pred, digits=4)
    accuracy = accuracy_score(y_test, y_pred)

    if printable:
        print(report)

    return report, accuracy


def centralized_learning(x_train, y_train, x_test, y_test, epochs, batch_size, dataset='mnist'):
    if dataset=='mnist':
        if use_cuda:
            model = CNN4FL_MNIST().cuda()
        else:
            model = CNN4FL_MNIST()
    elif dataset=='fmnist':
        if use_cuda:
            model = Lenet5().cuda()
        else:
            model = Lenet5()
    elif dataset=='cifar10':
        if use_cuda:
            # model = resnet20().cuda() # FL 세팅에서는 resnet32를, Central 상황에서는 resnet20으로 들어가있어 32로 통일하였습니다
            # model = resnet32().cuda()
            model = models.resnet18(pretrained=False)
            model.fc = nn.Linear(512, 10)
            model.cuda()
        else:
            # model = resnet20()
            # model = resnet32()
            model = models.resnet18(pretrained=False)
            model.fc = nn.Linear(512, 10)
            # model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    train_data, test_data = create_dataloader(x_train, y_train, x_test, y_test, batch_size, dataset)
    print("------ Centralized Model ------")
    logs = list()
    for epoch in range(epochs):
        adjust_learning_rate(0.05, optimizer, epoch)

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
    tempname = dataset + '_' + str(epochs) + '_' + str(batch_size) + '_' + str(central_test_accuracy)
    filename = '../data/exp_result/' + 'central_' + tempname + filetime + '.pkl'

    log_dict = {
        'main': logs
    }

    with open(filename, 'wb') as f:
        pickle.dump(log_dict, f)
    print('\n==> SAVE LOG:', filename, end='\n\n')

    return model


def compare_local_and_merged_model(main_model, local_model_dict,
                                   x_test_dict, y_test_dict,
                                   printable=True, dataset='mnist'):
    number_of_samples = len(local_model_dict)
    accuracy_table = pd.DataFrame(data=np.zeros((number_of_samples, 3)),
                                  columns=['local', 'local_ind_model', 'merged_main_model'])
    for i, (m, x, y) in enumerate(zip(local_model_dict, x_test_dict, y_test_dict)):
        local_model = local_model_dict[m]
        x_test = x_test_dict[x].float()
        y_test = y_test_dict[y]

        if dataset == 'cifar10':
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            test_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                normalize
            ])
            test_dataset = CustomTensorDataset(tensors=(x_test, y_test), transform=test_transform)

            test_data = torch.utils.data.DataLoader(test_dataset,
                                                    batch_size=len(x_test), shuffle=False)

            x_test = next(iter(test_data))[0]
            y_test = next(iter(test_data))[1]
        elif dataset == 'fmnist':
            transform = transforms.Compose(
                [transforms.ToPILImage(), transforms.Resize((35, 35)), transforms.ToTensor()])
            test_dataset = CustomTensorDataset(tensors=(x_test, y_test), transform=transform)

            test_data = torch.utils.data.DataLoader(test_dataset, batch_size=len(x_test), shuffle=False)

            x_test = next(iter(test_data))[0]
            y_test = next(iter(test_data))[1]

        y_pred = local_model(x_test).argmax(dim=1)
        local_accuracy = accuracy_score(y_pred.cpu(), y_test.cpu())

        y_pred = main_model(x_test).argmax(dim=1)
        main_accuracy = accuracy_score(y_pred.cpu(), y_test.cpu())

        accuracy_table.loc[i, 'local'] = 'local ' + str(i)
        accuracy_table.loc[i, 'local_ind_model'] = local_accuracy
        accuracy_table.loc[i, 'merged_main_model'] = main_accuracy

    if printable:
        print(accuracy_table)

    return accuracy_table


def save_model(model, path):
    path = path + '.pt' # torch weight 저장을 위한 .pt 확장자 추가
    print('==> SAVE MODEL: ', path)
    torch.save(model.state_dict(), path)


def load_model(path, dataset='mnist'):
    if dataset=='mnist':
        model = CNN4FL_MNIST()
        if use_cuda:
            model.cuda()
    elif dataset=='fmnist':
        model = Lenet5()
        if use_cuda:
            model.cuda()
    elif dataset=='cifar10':
        # model = resnet32()
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(512, 10)
        if use_cuda:
            model.cuda()
        
    model.load_state_dict(torch.load(path))

    return model


def do_centralize_learning(tr_X, tr_y, te_X, te_y, batch_size, epochs, dataset='mnist'):
    centralized_model = centralized_learning(
        tr_X, tr_y, te_X, te_y,
        epochs=epochs,
        batch_size=batch_size,
        dataset=dataset
    )

    create_eval_report(centralized_model, te_X, te_y, dataset=dataset)

    save_base = '../data/model_' + dataset
    if not os.path.exists(save_base):
        os.makedirs(save_base)
    
    model_name = os.path.join(save_base, 'centralized_model_' + str(epochs) + '_epochs')
    model_name += time.strftime("_%Y%m%d-%H%M%S")
    save_model(centralized_model, model_name)
    del centralized_model


def do_FL(_dataset, _iteration, _epochs, _batch_size,
          _tr_X_dict, _tr_y_dict, _te_X_dict, _te_y_dict, _te_X, _te_y,
          _num_of_local, _log_name, cur_cnt, total_cnt,
          uncert=0,
          dataset='mnist',
          uncert_threshold=0.2,
          verbose=False):
    print('\n===================================')
    print('CUDA:', use_cuda)
    print('MODEL: Federated Learning')
    print('DIST: {} ({}/{})'.format(_dataset, cur_cnt, total_cnt))
    print('DATASET:', dataset)
    print('ITERATION:', _iteration)
    print('EPOCHS:', _epochs)
    print('BATCH_SIZE:', _batch_size)
    print('LOG_NAME:', _log_name)
    print('===================================\n')

    if uncert == 2:
        main_model, local_models = uncert_federated_learning(
            _tr_X_dict, _tr_y_dict, _te_X_dict, _te_y_dict, _te_X, _te_y,
            _num_of_local,
            iteration=_iteration,
            epochs=_epochs,
            batch_size=_batch_size,
            log_name=_log_name,
            dataset=dataset,
            uncert_threshold=uncert_threshold,
            verbose=verbose
        )
    elif uncert == 1:
        main_model, local_models = federated_learning(
            _tr_X_dict, _tr_y_dict, _te_X_dict, _te_y_dict, _te_X, _te_y,
            _num_of_local,
            iteration=_iteration,
            epochs=_epochs,
            batch_size=_batch_size,
            log_name=_log_name,
            dataset=dataset,
            pre_train=True,
            verbose=verbose
        )
    else:
        main_model, local_models = federated_learning(
            _tr_X_dict, _tr_y_dict, _te_X_dict, _te_y_dict, _te_X, _te_y,
            _num_of_local,
            iteration=_iteration,
            epochs=_epochs,
            batch_size=_batch_size,
            log_name=_log_name,
            dataset=dataset,
            pre_train=False,
            verbose=verbose
        )

    create_eval_report(main_model, _te_X, _te_y, dataset=dataset)
    compare_local_and_merged_model(main_model, local_models,
                                   _te_X_dict, _te_y_dict, dataset=dataset)
    
    save_base = '../data/model_' + dataset

    if not os.path.exists(save_base):
        os.makedirs(save_base)
        
    model_name = os.path.join(save_base, _log_name + '_main_model')
    
    model_name += time.strftime("_%Y%m%d-%H%M%S")
    save_model(main_model, model_name)
    # Release variables
    del _tr_X_dict
    del _tr_y_dict
    del _te_X_dict
    del _te_y_dict
    del _te_X
    del _te_y

    return main_model


def do_non_corruption(tr_X, tr_y, te_X, te_y, batch_size, iteration, epochs, local_num, uncert_fedavg, mode='iid',
                      dataset='mnist'):
    if mode == 'iid':
        tr_X_dict, tr_y_dict, te_X_dict, te_y_dict = create_corrupted_iid_samples(
            tr_X, tr_y, te_X, te_y,
            cor_local_ratio=0.0,
            num_of_sample=local_num,
            verbose=True
        )
    else:
        tr_X_dict, tr_y_dict, te_X_dict, te_y_dict = create_corrupted_non_iid_samples(
            tr_X, tr_y, te_X, te_y,
            cor_local_ratio=0.0,
            num_of_sample=local_num,
            verbose=True
        )

    log_name = 'federated_' + mode + '_' + FL_ALGO[uncert_fedavg] + '_non_corrupted'

    model = do_FL(mode, iteration, epochs, batch_size,
                  tr_X_dict, tr_y_dict, te_X_dict, te_y_dict,
                  te_X, te_y, local_num, log_name,
                  1, 1,
                  dataset=dataset,
                  uncert=uncert_fedavg,
                  verbose=False)

    # Release variables
    del tr_X_dict
    del tr_y_dict
    del te_X_dict
    del te_y_dict
    del model


def do_iid_corruption(total_cnt, cur_cnt,
                      tr_X, tr_y, te_X, te_y,
                      batch_size, iteration, epochs, local_num, uncert_fedavg,
                      cor_local_ratio, cor_label_ratio, cor_data_ratio, cor_mode, dataset='mnist'):
    tr_X_dict, tr_y_dict, te_X_dict, te_y_dict = create_corrupted_iid_samples(
        tr_X, tr_y, te_X, te_y,
        cor_local_ratio=cor_local_ratio,
        cor_label_ratio=cor_label_ratio,
        cor_data_ratio=cor_data_ratio,
        mode=cor_mode,
        num_of_sample=local_num,
        verbose=True,
        dataset='mnist'
    )

    log_name = 'federated_' + 'iid' + '_'
    log_name += FL_ALGO[uncert_fedavg] + '_'
    log_name += str(int(cor_local_ratio * 10)) + '_cor_local_'
    log_name += str(int(cor_label_ratio * 100)) + '_cor_label_'
    log_name += CORRUPTION_MODE[cor_mode]

    model = do_FL('iid', iteration, epochs, batch_size,
                  tr_X_dict, tr_y_dict, te_X_dict, te_y_dict,
                  te_X, te_y, local_num, log_name,
                  cur_cnt, total_cnt,
                  dataset=dataset,
                  uncert=uncert_fedavg,
                  verbose=False)

    # Release variables
    del tr_X_dict
    del tr_y_dict
    del te_X_dict
    del te_y_dict
    del model


def do_iid_backdoor(total_cnt, cur_cnt,
                    tr_X, tr_y, te_X, te_y,
                    batch_size, iteration, epochs, local_num, uncert_fedavg,
                    cor_local_ratio, cor_label_ratio, cor_data_ratio, target_label, dataset='mnist'):
    tr_X_dict, tr_y_dict, te_X_dict, te_y_dict, val_X_dict, val_y_dict = create_backdoor_iid_samples(
        tr_X, tr_y, te_X, te_y, target_label=target_label,
        cor_local_ratio=cor_local_ratio,
        cor_label_ratio=cor_label_ratio,
        cor_data_ratio=cor_data_ratio,
        num_of_sample=local_num,
        verbose=True,
        dataset=dataset
    )

    log_name = 'federated_' + 'iid' + '_'
    log_name += FL_ALGO[uncert_fedavg] + '_'
    log_name += str(int(cor_local_ratio * 10)) + '_cor_local_'
    log_name += str(int(cor_label_ratio * 100)) + '_cor_label_'
    log_name += CORRUPTION_MODE[2]

    main_model = do_FL('iid', iteration, epochs, batch_size,
                       tr_X_dict, tr_y_dict, te_X_dict, te_y_dict,
                       te_X, te_y, local_num, log_name,
                       cur_cnt, total_cnt,
                       dataset=dataset,
                       uncert=uncert_fedavg,
                       verbose=False)

    asr = cal_asr(main_model, te_y_dict, val_X_dict, val_y_dict, target_label,
                  dataset=dataset)

    filetime = time.strftime("_%Y%m%d-%H%M%S")
    temp_name = '_' + dataset + '_' + str(local_num) + '_' + str(iteration) + '_' + str(epochs)
    temp_name += '_' + str(batch_size) + '_' + str(asr)

    filename = '../data/exp_result/' + log_name + temp_name + filetime + '.asr'

    with open(filename, 'wb') as f:
        pickle.dump(asr, f)

    # Release variables
    del tr_X_dict
    del tr_y_dict
    del te_X_dict
    del te_y_dict
    del val_X_dict
    del val_y_dict
    del main_model


def do_non_iid_corruption(total_cnt, cur_cnt,
                          tr_X, tr_y, te_X, te_y,
                          batch_size, iteration, epochs, local_num, uncert_fedavg,
                          cor_local_ratio, cor_minor_label_cnt, cor_major_data_ratio, cor_minor_data_ratio,
                          pdist, cor_mode, dataset='mnist'):
    tr_X_dict, tr_y_dict, te_X_dict, te_y_dict = create_corrupted_non_iid_samples(
        tr_X, tr_y, te_X, te_y,
        cor_local_ratio=cor_local_ratio,
        cor_minor_label_cnt=cor_minor_label_cnt,
        cor_major_data_ratio=cor_major_data_ratio,
        cor_minor_data_ratio=cor_minor_data_ratio,
        mode=cor_mode,
        pdist=pdist,
        num_of_sample=local_num,
        verbose=True,
        dataset=dataset
    )

    log_name = 'federated_' + 'non-iid' + '_'
    log_name += FL_ALGO[uncert_fedavg] + '_'
    log_name += str(int(cor_minor_label_cnt)) + '_cor_minor_label_'
    log_name += str(int(cor_minor_data_ratio * 100)) + '_cor_minor_data_'
    log_name += CORRUPTION_MODE[cor_mode]

    model = do_FL('non-iid', iteration, epochs, batch_size,
                  tr_X_dict, tr_y_dict, te_X_dict, te_y_dict,
                  te_X, te_y, local_num, log_name,
                  cur_cnt, total_cnt,
                  uncert=uncert_fedavg,
                  dataset=dataset,
                  uncert_threshold=0.1,
                  verbose=False)

    # Release variables
    del tr_X_dict
    del tr_y_dict
    del te_X_dict
    del te_y_dict
    del model


def do_non_iid_backdoor(total_cnt, cur_cnt, tr_X, tr_y, te_X, te_y,
                        batch_size, iteration, epochs, local_num, uncert_fedavg,
                        cor_local_ratio, cor_minor_label_cnt, cor_major_data_ratio,
                        cor_minor_data_ratio, pdist, target_label, dataset='mnist'):
    tr_X_dict, tr_y_dict, te_X_dict, te_y_dict, val_X_dict, val_y_dict = create_backdoor_non_iid_samples(
        tr_X, tr_y, te_X, te_y, target_label,
        cor_local_ratio=cor_local_ratio,
        cor_minor_label_cnt=cor_minor_label_cnt,
        cor_major_data_ratio=cor_major_data_ratio,
        cor_minor_data_ratio=cor_minor_data_ratio,
        pdist=pdist,
        num_of_sample=local_num,
        verbose=True,
        dataset=dataset
    )

    log_name = 'federated_' + 'non-iid' + '_'
    log_name += FL_ALGO[uncert_fedavg] + '_'
    log_name += str(int(cor_minor_label_cnt)) + '_cor_minor_label_'
    log_name += str(int(cor_minor_data_ratio * 100)) + '_cor_minor_data_'
    log_name += CORRUPTION_MODE[2]

    main_model = do_FL('non-iid', iteration, epochs, batch_size,
                       tr_X_dict, tr_y_dict, te_X_dict, te_y_dict,
                       te_X, te_y, local_num, log_name,
                       cur_cnt, total_cnt,
                       uncert=uncert_fedavg,
                       dataset=dataset,
                       uncert_threshold=0.1,
                       verbose=False)

    asr = cal_asr(main_model, te_y_dict, val_X_dict, val_y_dict, target_label,
                  dataset=dataset)

    filetime = time.strftime("_%Y%m%d-%H%M%S")
    temp_name = '_' + dataset + '_' + str(local_num) + '_' + str(iteration) + '_' + str(epochs)
    temp_name += '_' + str(batch_size) + '_' + str(asr)

    filename = '../data/exp_result/' + log_name + temp_name + filetime + '.asr'

    with open(filename, 'wb') as f:
        pickle.dump(asr, f)

    # Release variables
    del tr_X_dict
    del tr_y_dict
    del te_X_dict
    del te_y_dict
    del val_X_dict
    del val_y_dict
    del main_model
