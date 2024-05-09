import torch
import time
import sklearn.datasets
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# Загрузка данных
wine = sklearn.datasets.load_wine()
wine_data = wine.data[:, :2]  # Берем только первые два признака для простоты
wine_target = wine.target

# Преобразование данных в тензоры PyTorch
X = torch.FloatTensor(wine_data)
y = torch.LongTensor(wine_target)

# Разделение данных на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Определение модели
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

# Список размеров батчей для исследования
batch_sizes = [2, 4, 8, 16, 32, 64, 128]

# Словарь для сохранения времени обучения для каждого размера батча
training_times = {}

for batch_size in batch_sizes:
    # Создание DataLoader для обучения с текущим размером батча
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Инициализация модели
    wine_net = WineNet()

    # Определение функции потерь и оптимизатора
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(wine_net.parameters(), lr=0.01)

    # Обучение модели
    start_time = time.time()
    for epoch in range(100):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = wine_net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
    end_time = time.time()

    # Сохранение времени обучения
    training_times[batch_size] = end_time - start_time

# Построение графика зависимости времени обучения от размера батча
plt.plot(list(training_times.keys()), list(training_times.values()), marker='o')
plt.title('Зависимость времени обучения от размера батча')
plt.xlabel('Размер батча')
plt.ylabel('Время обучения (сек)')
plt.grid(True)
plt.savefig('График_зависимости_времени_обучения_от_размера_батча.png')
