import pandas as pd
import numpy as np
import joblib
import os

# Загрузка обученной модели
model = joblib.load('trained_model.pkl')

# Загрузка данных
data = pd.read_csv('df_hack_final.csv')

# Преобразование даты в datetime
data['datetime'] = pd.to_datetime(data['MEAS_DT'], errors='coerce')

# Убираем строки с NaN в целевой переменной
data.dropna(subset=['Ni_rec'], inplace=True)

# Выбираем признаки для модели
features = ['Cu_oreth', 'Ni_oreth', 'Mass_1', 'Dens_4', 'Vol_4', 'Cu_4F', 'Ni_4F', 'Ni_4.1C', 'Ni_4.2C', 'Ni_5F']
target = 'Ni_rec'

# Выбираем 20 временных интервалов (например, последние 20 интервалов)
time_intervals = data['datetime'].dropna().unique()[-20:]

# Папка для результатов
os.makedirs('test_results', exist_ok=True)

# Создаем список для тестовых данных
test_data = []

for timestamp in time_intervals:
    # Фильтруем данные по текущему временном интервалу
    current_data = data[data['datetime'] == timestamp]
    
    # Для каждого признака вычисляем минимальное и максимальное значение
    for fm_number in current_data['FM_4.1_A'].unique():  # Например, FM_4.1_A — номер ФМ
        for feature in features:
            # Фильтруем по номеру ФМ и признаку
            feature_data = current_data[current_data['FM_4.1_A'] == fm_number]
            min_value = feature_data[feature].min()
            max_value = feature_data[feature].max()
            
            # Формируем строку для test.csv
            test_data.append([f'Test {len(test_data) + 1}', timestamp.strftime('%Y.%m.%d %H:%M'), fm_number, feature, min_value, max_value])

# Преобразуем список в DataFrame
test_df = pd.DataFrame(test_data, columns=['Test', 'Datetime', 'FM', 'Feature', 'Min', 'Max'])

# Сохраняем в файл test.csv
test_df.to_csv('test_results/test.csv', index=False)

print("Результаты сохранены в 'test_results/test.csv'.")
