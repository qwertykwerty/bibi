import pandas as pd
import numpy as np
import os  # Для работы с директориями

# Пример функции для генерации границ
def generate_boundaries(start_value, min_step, num_periods):
    boundaries = []
    current_value = start_value
    for i in range(num_periods):
        if i > 0 and i % 8 == 0:  # Каждые 8 периодов границы могут изменяться
            current_value += min_step * np.random.randint(1, 3)  # случайное изменение на шаг
        boundaries.append(current_value)
    return boundaries

# Пример данных с границами для Ni_1.C
ni_c_min_start = 0.1
ni_c_max_start = 0.5
ni_c_min_step = 0.1
ni_c_max_step = 0.1
num_periods = 20

# Генерация для Ni_1.C
ni_c_min = generate_boundaries(ni_c_min_start, ni_c_min_step, num_periods)
ni_c_max = generate_boundaries(ni_c_max_start, ni_c_max_step, num_periods)

# Добавление других признаков по аналогии
cu_1_c_min = generate_boundaries(0.2, 0.1, num_periods)
cu_1_c_max = generate_boundaries(0.7, 0.1, num_periods)
cu_2_t_min = generate_boundaries(0.01, 0.01, num_periods)
cu_2_t_max = generate_boundaries(0.1, 0.01, num_periods)

# Пример для Ni_2.C
ni_2_c_min = generate_boundaries(0.2, 0.1, num_periods)
ni_2_c_max = generate_boundaries(0.6, 0.1, num_periods)

# Пример для Cu_3.C
cu_3_c_min = generate_boundaries(0.3, 0.1, num_periods)
cu_3_c_max = generate_boundaries(0.8, 0.1, num_periods)

# Пример для Cu_4.C
cu_4_c_min = generate_boundaries(0.4, 0.1, num_periods)
cu_4_c_max = generate_boundaries(0.9, 0.1, num_periods)

# Пример для Cu_5.T
cu_5_t_min = generate_boundaries(0.05, 0.01, num_periods)
cu_5_t_max = generate_boundaries(0.15, 0.01, num_periods)

# Создание DataFrame для 20 временных интервалов
df_test = pd.DataFrame({
    'Ni_1.C_min': ni_c_min,
    'Ni_1.C_max': ni_c_max,
    'Cu_1.C_min': cu_1_c_min,
    'Cu_1.C_max': cu_1_c_max,
    'Cu_2.T_min': cu_2_t_min,
    'Cu_2.T_max': cu_2_t_max,
    'Ni_2.C_min': ni_2_c_min,
    'Ni_2.C_max': ni_2_c_max,
    'Cu_3.C_min': cu_3_c_min,
    'Cu_3.C_max': cu_3_c_max,
    'Cu_4.C_min': cu_4_c_min,
    'Cu_4.C_max': cu_4_c_max,
    'Cu_5.T_min': cu_5_t_min,
    'Cu_5.T_max': cu_5_t_max
})

# Создание директории, если она не существует
output_dir = 'test_results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Сохранение в CSV
df_test.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

print("Файл test.csv был успешно сгенерирован и сохранен.")
