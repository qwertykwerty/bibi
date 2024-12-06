import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib  # Для сохранения модели

# Загрузка данных
data = pd.read_csv('df_hack_final.csv')

# Просмотр данных
print(data)

# Просмотр пропусков
print("Количество пропусков в каждом столбце:")
print(data.isnull().sum())

# Проверка типов данных
print("\nИнформация о типах данных:")
print(data.info())

# Статистическое описание данных
print("\nСтатистическое описание:")
print(data.describe())

# Гистограмма для 'Ni_rec'
plt.hist(data['Ni_rec'], bins=20, color='skyblue', edgecolor='black')
plt.title("Распределение извлечения никеля (Ni_rec)")
plt.xlabel("Извлечение никеля, %")
plt.ylabel("Частота")
plt.show()

# Проверка зависимостей между числовыми признаками
numeric_data = data.select_dtypes(include=['float64', 'int64'])
corr = numeric_data.corr()
print("\nКорреляция между признаками:")
print(corr)

# Корреляция с целевой переменной 'Ni_rec'
print("\nКорреляция с Ni_rec:")
print(corr['Ni_rec'].sort_values(ascending=False))

# Выбор признаков для модели
X = data[['Cu_oreth', 'Ni_oreth', 'Mass_1', 'Vol_4', 'FM_4.1_A']]
y = data['Ni_rec']

# Удаление строк с NaN в целевой переменной
data = data.dropna(subset=['Ni_rec'])  # Удаляем строки, где y (Ni_rec) == NaN

# Удаление строк с NaN в признаках
X = X.loc[data.index]  # Оставляем только те строки, которые соответствуют строкам, где нет NaN в y

# Заполнение пропусков в признаках медианой
X.fillna(X.median(), inplace=True)

# Проверка размерности X и y перед разделением
print("\nРазмерность X:", X.shape)
print("Размерность y:", y.shape)

# Обновление y, чтобы оно соответствовало удалённым строкам в data
y = y.loc[data.index]

# Проверка размерности после согласования индексов
print("\nРазмерность X и y после синхронизации:")
print("Размерность X:", X.shape)
print("Размерность y:", y.shape)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Проверка на NaN в целевой переменной после разделения
print("NaN в y_train:", y_train.isnull().sum())
print("NaN в X_train:", X_train.isnull().sum().sum())

# Обучение модели (используем случайный лес)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Предсказания на тестовой выборке
y_pred = model.predict(X_test)

# Проверка пропусков в тестовых данных и предсказаниях
print("\nКоличество пропусков в y_test:", y_test.isnull().sum())
print("Количество пропусков в y_pred:", np.isnan(y_pred).sum())

# Удаление строк с NaN в y_test и y_pred
mask = ~np.isnan(y_pred) & ~y_test.isnull()  # Маска для удаления NaN
y_test = y_test[mask]
y_pred = y_pred[mask]

# Вычисление RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("\nRMSE:", rmse)

# Вывод важности признаков
feature_importances = model.feature_importances_
print("\nВажность признаков:")
print(pd.Series(feature_importances, index=X.columns).sort_values(ascending=False))

# Сохранение модели в файл
joblib.dump(model, 'trained_model.pkl')  # Сохраняем модель

print("Модель обучена и сохранена как 'trained_model.pkl'.")
