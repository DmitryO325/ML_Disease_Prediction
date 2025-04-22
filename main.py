# Импортируемые библиотеки
import statistics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from scipy import stats
import warnings

# Считываем файл train.csv, удаляя последний столбец, так как он содержит NaN
DATA_PATH = 'dataset/Training.csv'
data = pd.read_csv(DATA_PATH).dropna(axis=1)
# print(data.info())
# print(data.describe())
# print(data[:10])

# Проверяем, сбалансирован ли набор данных или нет
disease_counts = data['prognosis'].value_counts()
temp_df = pd.DataFrame({
    'Disease': disease_counts.index,
    'Counts': disease_counts.values
})

# # Создаём ступенчатый график
# plt.figure(figsize=(18, 8))
# sns.barplot(x='Disease', y='Counts', data=temp_df)
# plt.xticks(rotation=90)
# plt.show()

# Преобразовываем целевые значения в числа, используя LabelEncoder
encoder = LabelEncoder()
data['prognosis'] = encoder.fit_transform(data['prognosis'])

# Делим выборку на обучающую и тестовую
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)


# print(f"Train: {X_train.shape}, {y_train.shape}")
# print(f"Test: {X_test.shape}, {y_test.shape}")


# Определяем метрику качества (долю правильных ответов) для кросс-валидации K-Fold
def cv_scoring(estimator, X, y):
    return accuracy_score(y, estimator.predict(X))


# Инициализируем модели
models = {
    'Метод опорных векторов': SVC(),
    'Наивный байесовский алгоритм': GaussianNB(),
    'Случайный лес': RandomForestClassifier(random_state=18)
}

# Получаем оценки кросс-валидации для моделей
for model_name in models:
    model = models[model_name]
    scores = cross_val_score(model, X, y, cv=10, n_jobs=-1, scoring=cv_scoring)

    # print('=='*30)
    # print(model_name)
    # print(f'Оценки: {scores}')
    # print(f'Средняя оценка: {np.mean(scores)}')

# Обучаем и тестируем классификатор на основе метода опорных векторов
svm_model = SVC()
svm_model.fit(X_train, y_train)
predict = svm_model.predict(X_test)

print(f'Доля правильных ответов (accuracy) на обучающей выборке с помощью SVM Classifier: '
      f'{accuracy_score(y_train, svm_model.predict(X_train)) * 100}%')

print(f'Доля правильных ответов на тестовой выборке с помощью SVM Classifier: '
      f'{accuracy_score(y_test, predict) * 100}%\n')

# print(f'Точность (precision) на обучающей выборке с помощью SVM Classifier: '
#       f'{precision_score(y_train, svm_model.predict(X_train), average='weighted') * 100}%')
#
# print(f'Точность на тестовой выборке с помощью SVM Classifier: '
#       f'{precision_score(y_test, predict, average='weighted') * 100}%\n')
#
# print(f'Полнота (recall) на обучающей выборке с помощью SVM Classifier: '
#       f'{recall_score(y_train, svm_model.predict(X_train), average='weighted') * 100}%')
#
# print(f'Полнота на тестовой выборке с помощью SVM Classifier: '
#       f'{recall_score(y_test, predict, average='weighted') * 100}%\n')

# cf_matrix = confusion_matrix(y_test, predict)
# plt.figure(figsize=(12, 8))
# sns.heatmap(cf_matrix, annot=True)
# plt.title('Матрица ошибок для SVM классификатора на тестовых данных')
# plt.show()

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
predict = nb_model.predict(X_test)

print(f'Доля правильных ответов (accuracy) на обучающей выборке с помощью NB Classifier: '
      f'{accuracy_score(y_train, nb_model.predict(X_train)) * 100}%')

print(f'Доля правильных ответов на тестовой выборке с помощью NB Classifier: '
      f'{accuracy_score(y_test, predict) * 100}%\n')

# print(f'Точность (precision) на обучающей выборке с помощью NB Classifier: '
#       f'{precision_score(y_train, nb_model.predict(X_train), average='weighted') * 100}%')
#
# print(f'Точность на тестовой выборке с помощью NB Classifier: '
#       f'{precision_score(y_test, predict, average='weighted') * 100}%\n')
#
# print(f'Полнота (recall) на обучающей выборке с помощью NB Classifier: '
#       f'{recall_score(y_train, nb_model.predict(X_train), average='weighted') * 100}%')
#
# print(f'Полнота на тестовой выборке с помощью NB Classifier: '
#       f'{recall_score(y_test, predict, average='weighted') * 100}%\n')

# cf_matrix = confusion_matrix(y_test, predict)
# plt.figure(figsize=(12, 8))
# sns.heatmap(cf_matrix, annot=True)
# plt.title('Матрица ошибок для NB классификатора на тестовых данных')
# plt.show()

rf_model = RandomForestClassifier(random_state=18)
rf_model.fit(X_train, y_train)
predict = rf_model.predict(X_test)

print(f'Доля правильных ответов (accuracy) на обучающей выборке с помощью Random Forest Classifier: '
      f'{accuracy_score(y_train, rf_model.predict(X_train)) * 100}%')

print(f'Доля правильных ответов на тестовой выборке с помощью Random Forest Classifier: '
      f'{accuracy_score(y_test, predict) * 100}%\n')

# print(f'Точность (precision) на обучающей выборке с помощью Random Forest Classifier: '
#       f'{precision_score(y_train, rf_model.predict(X_train), average='weighted') * 100}%')
#
# print(f'Точность на тестовой выборке с помощью Random Forest Classifier: '
#       f'{precision_score(y_test, predict, average='weighted') * 100}%\n')
#
# print(f'Полнота (recall) на обучающей выборке с помощью Random Forest Classifier: '
#       f'{recall_score(y_train, rf_model.predict(X_train), average='weighted') * 100}%')
#
# print(f'Полнота на тестовой выборке с помощью Random Forest Classifier: '
#       f'{recall_score(y_test, predict, average='weighted') * 100}%\n')

# cf_matrix = confusion_matrix(y_test, predict)
# plt.figure(figsize=(12, 8))
# sns.heatmap(cf_matrix, annot=True)
# plt.title('Матрица ошибок для RF классификатора на тестовых данных')
# plt.show()

final_svm_model = SVC()
final_nb_model = GaussianNB()
final_rf_model = RandomForestClassifier(random_state=18)
final_svm_model.fit(X, y)
final_nb_model.fit(X, y)
final_rf_model.fit(X, y)

# Считываем тестовые данные
test_data = pd.read_csv('dataset/Testing.csv').dropna(axis=1)

test_X = test_data.iloc[:, :-1]
test_y = encoder.transform(test_data.iloc[:, -1])

# Делаем прогнозы на основе моды прогнозов, которые дали классификаторы
svm_preds = final_svm_model.predict(test_X)
nb_preds = final_nb_model.predict(test_X)
rf_preds = final_rf_model.predict(test_X)

final_preds = [stats.mode([i, j, k])[0] for i, j, k in zip(svm_preds, nb_preds, rf_preds)]

print(f'Доля правильных ответов на тестовом наборе данных с помощью комбинированной модели: '
      f'{accuracy_score(test_y, final_preds) * 100}%')

# cf_matrix = confusion_matrix(test_y, final_preds)
# plt.figure(figsize=(12, 8))
# sns.heatmap(cf_matrix, annot=True)
# plt.title('Матрица ошибок для комбинированной модели на тестовом наборе данных')
# plt.show()

symptoms = X.columns.values

# Создадим словарь с индексами болезней, чтобы закодировать входные симптомы в числовую форму
symptom_index = {}

for index, value in enumerate(symptoms):
    symptom = ' '.join([element.capitalize() for element in value.split('_')])
    symptom_index[symptom] = index

data_dict = {
    'symptom_index': symptom_index,
    'predictions_classes': encoder.classes_
}


# Определяем функцию
# Входные данные: строки, содержащие симптомы, разделяемые запятыми
# Выходные данные: сгенерированные предсказания модели
def predict_disease(symptoms):
    symptoms = symptoms.split(',')

    # Создаём входные данные для модели
    input_data = [0] * len(data_dict['symptom_index'])

    for symptom in symptoms:
        index = data_dict['symptom_index'][symptom]
        input_data[index] = 1

    # Изменяем размерность входных данных и конвертируем их в подходящий формат для предсказаний модели
    input_data = np.array(input_data).reshape(1, -1)

    # Генерируем отдельные выходные данные
    rf_prediction = data_dict['predictions_classes'][final_rf_model.predict(input_data)[0]]
    nb_prediction = data_dict['predictions_classes'][final_nb_model.predict(input_data)[0]]
    svm_prediction = data_dict['predictions_classes'][final_svm_model.predict(input_data)[0]]

    # Делаем конечное предсказание с помощью моды всех предсказаний
    # Используем statistics.mode вместо scipy.stats.mode
    final_prediction = statistics.mode([rf_prediction, nb_prediction, svm_prediction])
    predictions = {
        'rf_model_prediction': rf_prediction,
        'nb_model_prediction': nb_prediction,
        'svm_model_prediction': svm_prediction,
        'final_prediction': final_prediction
    }

    return predictions


# Тестируем функцию
warnings.filterwarnings('ignore', category=UserWarning)
print(predict_disease('Itching,Skin Rash,Nodal Skin Eruptions'))
