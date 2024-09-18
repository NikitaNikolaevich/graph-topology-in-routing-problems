import numpy as np

def optimal_a_star(N):
    return 8.09 * N ** -0.48 * (1 - 19.4 / (4.8 * np.log(N) + 8.8))

def myround(x, base=10):
    return base * round(x / base)

# Общая функция для расчета теоретического ускорения
def theoretical_acceleration(x, N):
    k = x * (1 + np.log(x) / np.log(N)) + 46 / (N ** 0.72 * x ** 0.5) * (0.28 - np.log(x) / np.log(N) / 2)
    return 1 / k

# Функция для оптимизации ускорения
def optimized_acceleration(x, a, N):
    k = x * (1 + np.log(x) / np.log(N)) + a / x ** 0.5
    return 1 / k