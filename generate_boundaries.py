import pandas as pd
import numpy as np
import joblib  # Для загрузки модели

# Загрузка обученной модели
model = joblib.load('trained_model.pkl')

# Пример функции для генерации границ на основе предсказанных значений
def generate_boundaries(start_value, predicted_value, min_step, num_periods):
    boundaries = []
    current_value = start_value
    for i in range(num_periods):
        # Каждые 8 периодов границы могут изменяться на основе предсказанного значения
        if i > 0 and i % 8 == 0:
            current_value = predicted_value + min_step * np.random.randint(1, 3)
        boundaries.append(current_value)
    return boundaries

# Загрузка исходных данных для генерации предсказаний
data = pd.read_csv('df_hack_final.csv')

# Выбор тех же признаков, которые использовались при обучении модели
X = data[['Cu_oreth', 'Ni_oreth', 'Mass_1', 'Vol_4', 'FM_4.1_A']]

# Прогнозирование с использованием обученной модели
y_pred = model.predict(X)

# Генерация границ для предсказанных значений
num_periods = 20  # Пример количества периодов

# Для каждого предсказанного значения мы генерируем границы
ni_c_min = generate_boundaries(0.1, y_pred[0], 0.1, num_periods)
ni_c_max = generate_boundaries(0.5, y_pred[0], 0.1, num_periods)

cu_oreth_min = generate_boundaries(0.2, y_pred[1], 0.1, num_periods)
cu_oreth_max = generate_boundaries(0.7, y_pred[1], 0.1, num_periods)

ni_oreth_min = generate_boundaries(0.3, y_pred[2], 0.05, num_periods)
ni_oreth_max = generate_boundaries(0.8, y_pred[2], 0.05, num_periods)

mass_1_min = generate_boundaries(0.5, y_pred[3], 0.05, num_periods)
mass_1_max = generate_boundaries(1.0, y_pred[3], 0.05, num_periods)

vol_4_min = generate_boundaries(0.05, y_pred[4], 0.01, num_periods)
vol_4_max = generate_boundaries(0.2, y_pred[4], 0.01, num_periods)

fm_4_1_a_min = generate_boundaries(0.1, y_pred[5], 0.02, num_periods)
fm_4_1_a_max = generate_boundaries(0.5, y_pred[5], 0.02, num_periods)

# Создание DataFrame для 20 временных интервалов
df_test = pd.DataFrame({
    'Ni_1.C_min': ni_c_min,
    'Ni_1.C_max': ni_c_max,
    'Cu_1.C_min': cu_oreth_min,
    'Cu_1.C_max': cu_oreth_max,
    'Ni_2.C_min': ni_oreth_min,
    'Ni_2.C_max': ni_oreth_max,
    'Mass_1.C_min': mass_1_min,
    'Mass_1.C_max': mass_1_max,
    'Vol_4.C_min': vol_4_min,
    'Vol_4.C_max': vol_4_max,
    'FM_4.1_A_min': fm_4_1_a_min,
    'FM_4.1_A_max': fm_4_1_a_max
})

# Сохранение в CSV
df_test.to_csv('test_results/test.csv', index=False)

print("Файл test.csv был успешно сгенерирован и сохранен.")
