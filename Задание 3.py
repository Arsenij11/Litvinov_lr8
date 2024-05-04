import torch
import torch.nn as nn
import torch.optim as optim

# Создаем класс нейронной сети с возможностью настройки количества слоев и метода активации
class CustomNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, activation):
        super(CustomNN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        if num_layers > 1:
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers = nn.ModuleList(layers)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError("Activation function should be 'relu', 'sigmoid', or 'tanh'")
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        x = self.output_layer(x)
        return x

# Функция для обучения нейронной сети с заданными параметрами
def train_network(input_size, output_size, X_train, y_train, hidden_size, num_layers, activation):
    model = CustomNN(input_size, hidden_size, output_size, num_layers, activation)
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

# Создаем случайные данные для обучения
input_size = 10
output_size = 1
num_samples = 1000
X_train = torch.randn(num_samples, input_size)
y_train = torch.randn(num_samples, output_size)

# Исследуем зависимость точности от количества нейронов, количества слоев и метода активации
hidden_sizes = [20, 40, 60]
num_layers_list = [1, 2, 3]
activations = ['relu', 'sigmoid', 'tanh']

for hidden_size in hidden_sizes:
    for num_layers in num_layers_list:
        for activation in activations:
            print(f'Training with hidden size: {hidden_size}, num layers: {num_layers}, activation: {activation}')
            try:
                loss = train_network(input_size, output_size, X_train, y_train, hidden_size, num_layers, activation)
            except:
                print("Failed to train the network.")
                continue
