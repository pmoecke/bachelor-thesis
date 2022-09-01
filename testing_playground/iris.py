from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(nn.Linear(4, 2000),
                                    nn.ReLU(),
                                    nn.Linear(2000, 1000),
                                    nn.ReLU(),
                                    nn.Linear(1000, 300),
                                    nn.ReLU())
        self.fc = nn.Linear(300, 3)

    def forward(self, x):
        x = self.layers(x)
        x = self.fc(x)
        out = F.softmax(x, dim=1)
        return out


def get_data():
    data = pd.read_csv("data/Iris/Iris.csv", header=None)
    data.columns = ["sepal.l,", "sepal.w", "petal.l", "petal.w", "class"]
    # Need to convert string to integers corresponding to class
    # setosa -> 0, versicolor -> 1, virginica -> 2
    data.loc[(data["class"] == "Iris-setosa"), "class"] = 0
    data.loc[data["class"] == "Iris-versicolor", "class"] = 1
    data.loc[data["class"] == "Iris-virginica", "class"] = 2
    data["class"] = pd.to_numeric(data["class"])  # Change type of "class" column to numeric

    x_vals = data.iloc[:, 0:4]
    y_vals = data.iloc[:, 4]

    x_tensor = torch.from_numpy(x_vals.values)
    labels = torch.from_numpy(y_vals.values)
    labels = F.one_hot(labels, num_classes=3)

    return x_tensor, labels

def train(x_vals, labels, model, optimizer, train_loader):
    model.train()
    loss_fn = nn.CrossEntropyLoss()

    for data, label in train_loader:
        # Assume only one batch to be processed -> loop only iterated through once
        optimizer.zero_grad()
        output = model(data.float())
        loss = loss_fn(output, label.float())
        loss.backward()
        optimizer.step()

    # test(x_vals, labels, model, train_loader, True, 0)

def test(x_vals, labels, model, test_loader, is_train, epoch):
    model.eval()
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data.float())
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            target_idx = target.argmax(dim=1, keepdim=True)

            for i in range(target.size(0)):
                if (target_idx[i][0] == pred[i][0]):
                    correct += 1
    
    classification_accuracy = correct / labels.size(0)
    print(f"Epoch {epoch}: test acc is {classification_accuracy}")
    return classification_accuracy

def normalize(data):
    means = data.mean(dim=0, keepdim=True)
    stds = data.std(dim=0, keepdim=True)
    norm_data = (data - means) / (stds + 1e-6)
    return norm_data

def simulation(axs, index):
    x_vals, labels = get_data()

    # Create train/test split, we have 150 observations -> 120 for training, 30 for testing
    indices = torch.randperm(150)
    x_train = x_vals[indices[:120]]
    x_test = x_vals[indices[120:]]
    y_train = labels[indices[:120]]
    y_test = labels[indices[120:]]

    x_train = normalize(x_train)
    x_test = normalize(x_test)


    train_ds = TensorDataset(x_train, y_train)
    test_ds = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=20) 
    test_loader = DataLoader(test_ds, batch_size=30)

    model = Net().float()
    optimizer = optim.SGD(model.parameters(), lr = 0.02)
    test_accuracies = []
    epochs = 250

    for epoch in range(epochs):
        train(x_train, y_train, model, optimizer, train_loader)
        curr_accuracy = test(x_test, y_test, model, test_loader, False, epoch + 1)
        test_accuracies.append(curr_accuracy)

    # Plot development of test accuracy over epochs
    for param in model.parameters():
        print(param.grad.data.size())
        print(param.requires_grad)

    train_acc = np.array(test_accuracies)
    axs[index].plot(np.arange(1, epochs + 1), train_acc)
    

def run_simulation():
    nsim = 1
    f, axs = plt.subplots(1, 10, sharey=True)

    for i in range(nsim):
        simulation(axs, i)
    
    plt.show()

if __name__ == "__main__":
    run_simulation()