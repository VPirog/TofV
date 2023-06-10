import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

def factor_analysis(data, num_factors):
    # Вычисление ковариационной матрицы
    cov_matrix = np.cov(data, rowvar=False)

    # Вычисление собственных значений и собственных векторов
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Сортировка собственных значений и собственных векторов в порядке убывания
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Выбор первых num_factors собственных векторов
    factor_loadings = sorted_eigenvectors[:, :num_factors]

    # Вычисление объясненной дисперсии
    explained_variance = sorted_eigenvalues[:num_factors] / np.sum(sorted_eigenvalues)

    return factor_loadings, explained_variance

# Загрузка набора данных "Ирисы Фишера"
iris = load_iris()
X = iris.data

factors = 2

# Вызов функции факторного анализа для 2 факторов
factor_loadings, explained_variance = factor_analysis(X, num_factors=factors)

# Вывод факторных нагрузок и объясненной дисперсии
print("Факторные нагрузки (без использования библиотеки):")
print(factor_loadings)
print("Объясненная дисперсия (без использования библиотеки):")
print(explained_variance)

# Применение PCA для редукции размерности до двух факторов
pca = PCA(n_components=factors)
X_pca = pca.fit_transform(X)

# Вывод факторных нагрузок и объясненной дисперсии с использованием PCA
print("Факторные нагрузки (с использованием PCA):")
print(pca.components_.T)
print("Объясненная дисперсия (с использованием PCA):")
print(pca.explained_variance_ratio_)
