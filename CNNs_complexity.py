import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from hvp_operator import to_vector, ModelHessianOperator
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from lanczos_hvp import lanczos
import time


if __name__ == '__main__':
    batch_size = 200

    train_dataset = MNIST(root='MNIST', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = MNIST(root='MNIST', train=False, transform=transforms.ToTensor(), download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    n_iters = 2000
    epochs = n_iters / (len(train_dataset) / batch_size)
    input_dim = 784
    output_dim = 10
    lr_rate = 0.001


    def create_model(a, b, c):
        return nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3),  # 28 - 2 = 26
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3),  # 26 - 2 = 24
            nn.ReLU(),

            nn.MaxPool2d(2),  # 24 / 2 = 12

            nn.Conv2d(8, 16, kernel_size=3),  # 12 - 2 = 10
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3),  # 10 - 2 = 8
            nn.ReLU(),

            nn.MaxPool2d(2),  # 8 / 2 = 4
            nn.Flatten(),  # nchannels * m * m = 16 * 4 * 4 = 256
            nn.Linear(256, a),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(a, b),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(b, c),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(c, 10),
        )


    a_lst = np.arange(25, 100, 5)
    b_lst = a_lst - 5
    c_lst = b_lst - 5

    time_spent_lst = []
    params_num_lst = []

    for i in range(len(a_lst)):
        a = a_lst[i]
        b = b_lst[i]
        c = c_lst[i]

        print('------------ {}/{} model -------------'.format(i + 1, len(a_lst)))
        model = create_model(a, b, c)
        print('a = {}, b = {}, c = {}'.format(a, b, c))
        params_num = sum(p.numel() for p in model.parameters())
        print('parameters number: {}'.format(params_num))
        params_num_lst.append(params_num)

        # model = model.cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate)

        lr_rate = 0.00007
        iter_num = 0
        for epoch in range(int(epochs)):
            for i, (images, labels) in enumerate(train_loader):
                # images = images.cuda()
                # labels = labels.cuda()
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                iter_num += 1
                if iter_num % 500 == 0:
                    # calculate Accuracy
                    correct = 0
                    total = 0
                    for images, labels in test_loader:
                        # images = images.cuda()
                        # labels = labels.cuda()
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        # for gpu, bring the predicted and labels back to cpu fro python operations to work
                        correct += (predicted == labels).sum()
                    accuracy = 100 * correct / total
                    print("Iteration: {}. Loss: {}. Accuracy: {}%.".format(iter_num, loss.item(), accuracy))

        test_batch = next(iter(test_loader))
        data_input = test_batch[0]
        data_target = test_batch[1]
        # op = ModelHessianOperator(model, criterion, data_input.cuda(), data_target.cuda())
        op = ModelHessianOperator(model, criterion, data_input, data_target)
        size = to_vector(model.parameters()).shape[0]
        print('The model has been trained')

        num_lanczos_vectors = int(0.5 * size)
        print('Starting Lanczoc method to find {} vectors'.format(num_lanczos_vectors))
        start = time.time()
        T, V = lanczos(operator=op, num_lanczos_vectors=num_lanczos_vectors, size=size, use_gpu=False)
        end = time.time()
        print(f'Time spent: {end - start}')
        time_spent_lst.append(end - start)