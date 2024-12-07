import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib  # Для сохранения модели

# Загрузка данных
data = pd.read_csv('df_hack_final.csv')

# Проверка колонок в датасете
print("Колонки в датасете:", data.columns)


# Преобразование даты и времени 

data['datetime'] = pd.to_datetime(data['MEAS_DT'], errors='coerce')


# Заполнение пропусков только в числовых данных
# Исключаем столбцы с датами и категориальными данными
numerical_data = data.select_dtypes(include=['float64', 'int64'])

# Заполнение пропусков медианой в числовых колонках
numerical_data.fillna(numerical_data.median(), inplace=True)

# Обновление данных в оригинальном DataFrame
data[numerical_data.columns] = numerical_data

# Выбор признаков для модели (измените на нужные вам признаки)
X = data[['Cu_oreth', 'Ni_oreth', 'Mass_1', 'Dens_4', 'Vol_4', 'Cu_4F', 'Ni_4F', 'Ni_4.1C', 'Ni_4.2C', 'Ni_5F']]
y = data['Ni_rec']

# Удаление строк с NaN в целевой переменной
data = data.dropna(subset=['Ni_rec'])  # Удаляем строки, где y (Ni_rec) == NaN
X = X.loc[data.index]  # Оставляем только те строки, которые соответствуют строкам, где нет NaN в y

# Проверка размерности X и y перед разделением
print("\nРазмерность X:", X.shape)
print("Размерность y:", y.shape)

# Обновление y, чтобы оно соответствовало удалённым строкам в data
y = y.loc[data.index]

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
