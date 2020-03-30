# /usr/local/bin/python3.7

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from tqdm import trange, tqdm
from hvp_operator import to_vector
from sampling import generate_weights, set_model_parameters
from torchvision.datasets import MNIST
import torchvision.transforms as transforms


def sample_scores(model, criterion, data_input, data_target, test_loader, weights_mle, N_samples=10, arbitrary=False, k=2):
    print(f'data_input.size() = {data_input.size()}')
    acc = []
    los = []
    for i in trange(N_samples):
        sample_weights = generate_weights(model, criterion, data_input, data_target, weights_mle, arbitrary=arbitrary, k=k)
        set_model_parameters(model, sample_weights)
        loss = criterion(model(data_input), data_target)
        los.append(loss.item())
        total = 0
        correct = 0

        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # for gpu, bring the predicted and labels back to cpu fro python operations to work
            correct += (predicted == labels).sum()
        accuracy = 100. * correct / float(total)
        acc.append(accuracy)
        set_model_parameters(model, weights_mle)
    return acc, los


if __name__ == '__main__':

    epochs = 5
    batch_size = 64
    lr = 5e-4
    class1_id = 1
    class2_id = 7

    train_dataset = MNIST(root='MNIST', train=True, transform=transforms.ToTensor(), download=True)
    for i in range(10):
        idx = (train_dataset.targets == i)
        if (i == 0) or (i % 2 == 0):
            train_dataset.targets[idx] = 0
        else:
            train_dataset.targets[idx] = 1
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = MNIST(root='MNIST', train=False, transform=transforms.ToTensor(), download=True)
    for i in range(10):
        idx = (test_dataset.targets == i)
        if (i == 0) or (i % 2 == 0):
            test_dataset.targets[idx] = 0
        else:
            test_dataset.targets[idx] = 1
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3),
        nn.ReLU(),
        nn.Conv2d(8, 8, kernel_size=3),
        nn.ReLU(),

        nn.MaxPool2d(2),

        nn.Conv2d(8, 16, kernel_size=3),
        nn.ReLU(),
        nn.Conv2d(16, 16, kernel_size=3),
        nn.ReLU(),

        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(256, 2),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_losses = []

    for epoch in trange(epochs):
        # Training
        train_losses.append([])
        model.train()
        for iter_num, (images, labels) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_losses[epoch].append(loss.item())

        print(f'\nTrain loss at epoch #{epoch} = {np.mean(train_losses[epoch]):.4f}\n')

        # Evaluation
        test_losses.append([])
        correct = 0
        total = 0
        model.eval()
        for iter_num_test, (images_test, labels_test) in enumerate(tqdm(test_loader)):
            with torch.no_grad():
                outputs_test = model(images_test)
                loss_test = criterion(outputs_test, labels_test)
            test_losses[epoch].append(loss_test.item())
            _, predicted = torch.max(outputs_test.data, 1)
            total += len(labels_test)
            correct += (predicted == labels_test).sum()
        accuracy_test = 100 * correct / total
        # print(f'outputs_test = {outputs_test}, labels_test = {labels_test}')
        print(f"\nTest loss at epoch #{epoch} = {np.mean(test_losses[epoch]):.4f}, "
              f"test accuracy: {accuracy_test:.2f}%\n")

    # Tuned weights
    weights_mle = to_vector(model.parameters()).detach().numpy()

    # Test dataset
    test_batch = next(iter(test_loader))
    data_input = test_batch[0]
    data_target = test_batch[1]

    scores = sample_scores(model, criterion, data_input, data_target, test_loader, weights_mle, N_samples=100, k=8)
    print(f'Accuracy mean = {np.mean(scores[0])}')
    print(f'Accuracy std = {np.std(scores[0])}')
    plt.hist(scores[0], bins=25)
    # plt.xlim(96, 100)
    plt.xlabel('accuracy')
    plt.ylabel('count')
    plt.savefig('experiments/acc_cnn_bin_reg.png')
    plt.close()

    print(f'Loss mean = {np.mean(scores[1])}')
    print(f'Loss std = {np.std(scores[1])}')

    plt.hist(scores[1], bins=25)
    # plt.xlim(0, 0.15)
    plt.xlabel('loss')
    plt.ylabel('count')
    plt.savefig('experiments/loss_cnn_bin_reg.png')
    plt.close()

