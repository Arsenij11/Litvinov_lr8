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

# Base Rate
base_rate = len(wine.target[wine.target == 1]) / len(wine.target)
print("Base Rate для датасета вин:", base_rate)

# Перебор значений test_size
test_sizes = np.arange(0.1, 1.0, 0.01)
accuracies = []

for test_size in test_sizes:
    X_train, X_test, y_train, y_test = train_test_split(
        wine_data[:, :2], wine_target, test_size=test_size, shuffle=True, random_state=42)

    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    class WineNet(torch.nn.Module):
        def __init__(self):
            super(WineNet, self).__init__()
            self.fc1 = torch.nn.Linear(2, 40)
            self.activ1 = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(40, 40)
            self.activ2 = torch.nn.ReLU()
            self.fc3 = torch.nn.Linear(40, 3)
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

    wine_net = WineNet()
    optimizer = torch.optim.Adam(wine_net.parameters(), lr=0.01)
    loss = torch.nn.CrossEntropyLoss()

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
    accuracies.append(accuracy)

    if accuracy < base_rate:
        print(f'При значении {accuracy} test_size предсказывает хуже, чем Base Rate')
    else:
        print(f"Test Size: {test_size}, Точность: {accuracy}")

plt.plot(test_sizes, accuracies, marker='o')
plt.axhline(y=base_rate, color='r', linestyle='--', label='Base Rate')
plt.title('Accuracy vs Test Size')
plt.xlabel('Test Size')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('График_зависимости_точности_от_test_size.png')
