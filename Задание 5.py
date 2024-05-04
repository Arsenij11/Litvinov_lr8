import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time


# Создаем простую нейронную сеть
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Генерируем случайные данные для обучения
input_size = 10
output_size = 1
num_samples = 1000
X_train = torch.randn(num_samples, input_size)
y_train = torch.randn(num_samples, output_size)

# Создаем DataLoader для обучения сети
dataset = TensorDataset(X_train, y_train)

# Задаем разные размеры батча для исследования
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

# Исследуем зависимость времени обучения от размера батча
for batch_size in batch_sizes:
    print(f'Training with batch size: {batch_size}')
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleNN(input_size, 40, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    start_time = time.time()
    for epoch in range(100):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    end_time = time.time()
    print(f'Time taken: {end_time - start_time} seconds')
