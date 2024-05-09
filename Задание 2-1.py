import torch
import random
import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split

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
    def __init__(self, n_hidden_neurons):
        super(WineNet, self).__init__()
        self.fc1 = torch.nn.Linear(2, n_hidden_neurons)
        self.activ1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(n_hidden_neurons, n_hidden_neurons)
        self.activ2 = torch.nn.Sigmoid()
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


unsatisfactory_accuracy_threshold = 0.7  # Порог для неудовлетворительной точности
check_num = 1
# Перебираем значения n_hidden_neurons от 1 до 40
for n_hidden_neurons in range(1, 41):
    print(f"Тест № {check_num}\nЗначение n_hidden_neurons={n_hidden_neurons}")
    wine_net = WineNet(n_hidden_neurons)
    check_num += 1
    optimizer = torch.optim.SGD(wine_net.parameters(), lr=0.01)
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
    accuracy = (test_preds == y_test).float().mean()
    print(f"Метрика: {accuracy.item()}")

    if accuracy < unsatisfactory_accuracy_threshold:
        print(f"Неудовлетворительная метрика при n_hidden_neurons={n_hidden_neurons}")
        continue

