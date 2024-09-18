import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from formulas import (
    optimal_a_star,
    myround,
    theoretical_acceleration,
    optimized_acceleration,
    )

# Функция для форматирования подписей
def format_labels(points_results, alpha_threshold=None):
    if alpha_threshold:
        return ['{:.2f}'.format(myround(p.alpha * 1000) / 1000) for p in points_results if p.alpha <= alpha_threshold]
    else:
        return ['{:.2f}'.format(myround(p.alpha * 1000) / 1000) for p in points_results]

# Функция для отрисовки графика ускорения
def plot_acceleration(ax, alpha, speed_up, N, a0, func, max_alpha):
    alpha1 = np.linspace(1 / N, max_alpha, 1000)
    ax.plot(alpha1, func(alpha1, N), '--', label='Theoretical acceleration', markersize=30, linewidth=8)
    idx = np.argwhere(alpha <= max_alpha)
    ax.plot(alpha[idx], speed_up[idx], 'o', label='Real acceleration', markersize=20)
    ax.axvline(x=a0, ymin=0, c='r', linestyle=':', label=r'$\alpha^*$', linewidth=6)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\gamma$')
    ax.legend()

# Функция для отрисовки боксплота ошибок
def plot_boxplot(ax, points_results, labels):
    boxprops = {"color": "black", "linewidth": 3, "facecolor": "#35478C"}
    medianprops = {"color": "r", "linewidth": 4}
    whiskerprops = {"color": "black", "linewidth": 3}
    capprops = {"color": "black", "linewidth": 3}
    
    errors = [np.array(p.errors) * 100 for p in points_results]
    ax.boxplot(errors, labels=labels, showfliers=False, vert=True, patch_artist=True,
               medianprops=medianprops, boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops)
    ax.tick_params(axis='x', labelrotation=60)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('error, %')

def theoretical_max_acceleration_nodes(nodes):
    return 1 / 13 * nodes ** 0.48

def theoretical_max_acceleration_density(density):
    return 1 / 13 * (2.8 / density) ** 0.48

def theoretical_alpha_star_nodes(nodes):
    return 8.09 * nodes ** -0.48 * (1 - 19.4 / (4.8 * np.log(nodes) + 8.8))

def theoretical_alpha_star_density(density):
    return 8.09 * (2.8 / density) ** -0.48 * (1 - 19.4 / (4.8 * np.log(2.8 / density) + 8.8))

def plot_loglog(x_data, y_data, theoretical_curve, x_label, y_label, legend_real, legend_theoretical, y_limits=None, x_limits=None):
    plt.figure(figsize=(15, 15))
    plt.loglog(x_data, y_data, 'o', label=legend_real, markersize=10, linewidth=8, alpha=0.5)
    
    # Сортируем данные по оси X для теоретической кривой
    x_sorted = np.sort(x_data)
    plt.loglog(x_sorted, theoretical_curve(x_sorted), '--', c='r', label=legend_theoretical, markersize=30, linewidth=8)
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    if y_limits:
        plt.ylim(y_limits)
    if x_limits:
        plt.xlim(x_limits)
    
    plt.legend()
    plt.show()

def plot_city_results(res, N, max_alpha=0.5, alpha_threshold=0.5):
    fig, axs = plt.subplots(2, 1)
    fig.set_figwidth(15)
    fig.set_figheight(15)

    alpha = np.array([p.alpha for p in res.points_results])
    speed_up = np.array([p.speed_up[0] for p in res.points_results])

    # Теоретическое значение alpha*
    a0 = optimal_a_star(N)
    
    # Построение графика ускорения
    plot_acceleration(axs[0], alpha, speed_up, N, a0, theoretical_acceleration, max_alpha)

    # Форматирование лейблов для боксплота
    labels = format_labels(res.points_results, alpha_threshold)

    # Построение боксплота ошибок
    plot_boxplot(axs[1], res.points_results, labels)

    fig.suptitle(fr'$N_0$={N}')
    plt.show()