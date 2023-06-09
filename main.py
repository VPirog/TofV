import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer

# Загрузка данных
data = pd.read_csv('rosstat.csv', delimiter=',')
# print(data)

# Выбор столбцов без значений null
st = ['Численность населения', 'Заболевание алкоголизмом', 'Заболевание наркоманией', 'Реальные доходы', 'Бедность', 'Образование детей', 'ВВП']

# Создание объекта для факторного анализа
factor_analysis = FactorAnalyzer(n_factors=4, rotation='quartimax')

# Применение модели факторного анализа к данным
factor_analysis.fit(data)


# Получение факторных нагрузок          
factor_loadings = factor_analysis.loadings_
factor_variances = factor_analysis.get_factor_variance()

# Округление факторных нагрузок до 6 знаков после запятой
rounded_loadings = [[round(val, 6) for val in loadings] for loadings in factor_loadings]

# Вывод округленных факторных нагрузок
print("Факторные нагрузки:")
for i, loadings in enumerate(rounded_loadings):
    row_formatted = [f"{value:<12}" for value in loadings]
    # print(f'{st[i]:<12}' + " ".join(row_formatted))
    print(f'{st[i]:<30}' + ",".join(row_formatted))

# Округление дисперсий факторов до 6 знаков после запятой
rounded_variances = tuple(np.round(val, 6) for val in factor_variances)

# Вывод округленных дисперсий факторов
print("\nДисперсии факторов:")
st = ['Дисперсия', 'Пропорциональная дисперсия', 'Накопленная дисперсия']
for i, loadings in enumerate(rounded_variances):
    row_formatted = [f"{value:<12}" for value in loadings]
    print(f'{st[i]:<30}' + ',' + ",".join(row_formatted))
