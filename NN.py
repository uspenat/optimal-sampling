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

if __name__ == '__main__':

    epochs = 5
    batch_size = 64
    lr = 5e-4

    train_dataset = MNIST(root='MNIST', train=True, transform=transforms.ToTensor(), download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = MNIST(root='MNIST', train=False, transform=transforms.ToTensor(), download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
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