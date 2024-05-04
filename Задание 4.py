from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.svm import SVC

# Загрузка датасета
wine = load_wine()

# Вычисление Base Rate для датасета о вине
base_rate = len(wine.target[wine.target == 1]) / len(wine.target)

# Создание тренировочного и тестового наборов данных с разными значениями test_size
test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]

for test_size in test_sizes:
    # Разделение данных на тренировочный и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=test_size, random_state=42)

    # Инициализация и обучение модели SVM
    model = SVC()
    model.fit(X_train, y_train)

    # Вычисление точности модели
    accuracy = model.score(X_test, y_test)

    # Сравнение точности модели с Base Rate
    if accuracy < base_rate:
        print(f"With test size {test_size}, the model predicts worse than Base Rate.")
    else:
        print(f"With test size {test_size}, the model predicts as good or better than Base Rate.")

print(f"The Base Rate for the wine dataset is: {base_rate}")
