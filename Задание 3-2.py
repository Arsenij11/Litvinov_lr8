import torch
import random
import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

wine = sklearn.datasets.load_wine()
wine_data = wine.data
wine_target = wine.target

X_train, X_test, y_train, y_test = train_test_split(
    wine_data[:, :2], wine_target, test_size=0.3, shuffle=True)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)


class WineNet(torch.nn.Module):
    def __init__(self, n_hidden_neurons, activation_func):
        super(WineNet, self).__init__()
        self.fc1 = torch.nn.Linear(2, n_hidden_neurons)
        self.activ1 = activation_func
        self.fc2 = torch.nn.Linear(n_hidden_neurons, n_hidden_neurons)
        self.activ2 = activation_func
        self.fc3 = torch.nn.Linear(n_hidden_neurons, 3)
        self.sm = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activ1(x)
        x = self.fc2(x)
        x = self.activ2(x)
        x = self.fc3(x)
        return x

    def inference(self, x):
        x = self.forward(x)
        x = self.sm(x)
        return x


# Перебор значений количества слоев
num_layers_values = [1, 2, 3]
accuracies_num_layers = []
for num_layers in num_layers_values:
    print(f'Номер слоя: {num_layers}')
    wine_net = WineNet(40, torch.nn.Sigmoid())  # Фиксируем количество нейронов в скрытом слое

    # Добавляем скрытые слои
    for _ in range(num_layers - 1):
        wine_net.add_module(f"fc{len(wine_net._modules) + 1}", torch.nn.Linear(40, 40))
        wine_net.add_module(f"activ{len(wine_net._modules) + 1}", torch.nn.Sigmoid())

    optimizer = torch.optim.SGD(wine_net.parameters(), lr=0.01)
    loss = torch.nn.CrossEntropyLoss()

    accuracies = []
    batch_size = 10
    for epoch in range(5000):
        order = np.random.permutation(len(X_train))
        for start_index in range(0, len(X_train), batch_size):
            optimizer.zero_grad()
            batch_indexes = order[start_index:start_index + batch_size]
            x_batch = X_train[batch_indexes]
            y_batch = y_train[batch_indexes]
            preds = wine_net.forward(x_batch)
            loss_value = loss(preds, y_batch)
            loss_value.backward()
            optimizer.step()

    test_preds = wine_net.forward(X_test)
    test_preds = test_preds.argmax(dim=1)
    accuracy = (test_preds == y_test).float().mean().item()
    accuracies_num_layers.append(accuracy)
    print(f'Точность: {accuracy}')

plt.figure(figsize=(10, 6))
plt.plot(num_layers_values, accuracies_num_layers, marker='o')
plt.title('Accuracy vs Number of Layers')
plt.xlabel('Number of Layers')
plt.ylabel('Accuracy')
plt.grid(True)
plt.savefig('График_зависимости_точности_распознавания_от_количества_слоёв.png')