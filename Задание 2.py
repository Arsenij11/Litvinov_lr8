import torch
import torch.nn as nn
import torch.optim as optim


# Создаем простой класс нейронной сети
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

# Генерируем случайные данные для обучения нейронной сети
input_size = 10
output_size = 1
num_samples = 1000
X_train = torch.randn(num_samples, input_size)
y_train = torch.randn(num_samples, output_size)

# Функция для обучения нейронной сети с заданным числом нейронов в скрытом слое
def train_network(n_hidden_neurons):
    model = SimpleNN(input_size, n_hidden_neurons, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch [{epoch+1}/100], Loss: {loss.item()}')

    return loss.item()

# Исследуем поведение сети при различных значениях числа нейронов в скрытом слое
for n_hidden_neurons in range(10, 100, 10):
    print(f'Training with {n_hidden_neurons} hidden neurons:')
    try:
        loss = train_network(n_hidden_neurons)
    except:
        print("Failed to train the network.")
        break
