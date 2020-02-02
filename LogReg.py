# /usr/local/bin/python3.7

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch.utils import data
from tqdm import trange
from hvp_operator import to_vector
from sampling import generate_weights, set_model_parameters


class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, X, y):
        'Initialization'
        self.labels = y
        self.vectors = X

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.vectors)

    def __getitem__(self, index):
        'Generates one sample of data'
        X = self.vectors[index]
        y = self.labels[index]
        return X.astype(np.float32), torch.tensor(y)


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs


def sample_scores(model, criterion, data_input, data_target, test_loader, weights_mle, N_samples=10, arbitrary=False, k=2):
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
    X, y = make_classification(
        n_samples=5000, n_features=20, n_redundant=0,
        n_informative=20, n_classes=2, random_state=42,
        n_clusters_per_class=1
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    pca = PCA(n_components=2)
    pca.fit(X)
    rX = pca.transform(X)

    plt.scatter(rX[:, 0], rX[:, 1], c=y)
    plt.savefig('experiments/dataset_pca.png')
    plt.close()

    batch_size = 200

    train_dataset = Dataset(X_train, y_train)
    test_dataset = Dataset(X_test, y_test)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    n_iters = 1000
    epochs = n_iters / (len(train_dataset) / batch_size)
    input_dim = 20
    output_dim = 2
    lr = 0.001

    model = LogisticRegression(input_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    iter_num = 0
    for epoch in range(int(epochs)):
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            iter_num += 1
            if iter_num % 10 == 0:
                # calculate Accuracy
                correct = 0
                total = 0
                for images, labels in test_loader:
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    # for gpu, bring the predicted and labels back to cpu from python operations to work
                    correct += (predicted == labels).sum()
                accuracy = 100 * correct / total
                print("Iteration: {}. Loss: {}. Accuracy: {}%.".format(iter_num, loss.item(), accuracy))

    # Tuned weights
    weights_mle = to_vector(model.parameters()).detach().numpy()

    # Test dataset
    test_batch = next(iter(test_loader))
    data_input = test_batch[0]
    data_target = test_batch[1]

    scores = sample_scores(model, criterion, data_input, data_target, test_loader, weights_mle, N_samples=100)

    plt.hist(scores[0], bins=20)
    plt.xlabel('accuracy')
    plt.ylabel('count')
    plt.savefig('experiments/acc_logreg_bin.png')
    plt.close()

    plt.hist(scores[1], bins=25)
    plt.xlabel('loss')
    plt.ylabel('count')
    plt.savefig('experiments/loss_logreg_bin.png')
    plt.close()
