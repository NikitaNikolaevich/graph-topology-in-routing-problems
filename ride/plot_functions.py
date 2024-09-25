import numpy as np
from matplotlib import pyplot as plt

from ride.formulas import (
    optimal_a_star,
    myround,
    theoretical_acceleration,
    )

# Функция для форматирования подписей
def format_labels(points_results, alpha_threshold=None):
    """
    Format the alpha values of a list of points results as labels.

    Parameters
    ----------
    points_results : list
        A list of points results, where each point result has an `alpha` attribute.
    alpha_threshold : float, optional
        An optional threshold for filtering the points results by alpha value (default is None).

    Returns
    -------
    list[str]
        A list of formatted alpha values as strings, rounded to two decimal places.
    """
    if alpha_threshold:
        return ['{:.2f}'.format(myround(p.alpha * 1000) / 1000) for p in points_results if p.alpha <= alpha_threshold]
    else:
        return ['{:.2f}'.format(myround(p.alpha * 1000) / 1000) for p in points_results]

# Функция для отрисовки графика ускорения
def plot_acceleration(ax, alpha, speed_up, N, a0, func, max_alpha=1):
    """
    Plot the acceleration curve for a given set of alpha values and a theoretical function.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The matplotlib axes to plot on.
    alpha : numpy.ndarray
        The alpha values to plot.
    speed_up : numpy.ndarray
        The corresponding speed-up values to plot.
    N : int
        The number of nodes.
    a0 : float
        The optimal alpha value.
    func : callable
        The theoretical function to plot.
    max_alpha : float, optional
        The maximum alpha value to plot (default is 1).

    Returns
    -------
    None
    """
    alpha1 = np.linspace(1 / N, max_alpha, 1000)
    ax.plot(alpha1, func(alpha1, N), '--', label='Theoretical acceleration', markersize=20, linewidth=5)
    idx = np.argwhere(alpha <= max_alpha)
    ax.plot(alpha[idx], speed_up[idx], 'o', label='Real acceleration', markersize=20)
    ax.axvline(x=a0, ymin=0, c='r', linestyle=':', label=r'$\alpha^*$', linewidth=5)
    ax.set_xlabel(r'$\frac{\text{number of clusters}}{\text{number of nodes}}$, $\alpha$')
    ax.set_ylabel(r'Acceleration, $\gamma$')
    ax.legend()

# Функция для отрисовки боксплота ошибок
def plot_boxplot(ax, points_results, labels):
    """
    Plot a boxplot of errors for a list of points results.

    Parameters
    ----------
    ax : matplotlib Axes
        The axes to plot on.
    points_results : list
        A list of points results, where each point result has an `errors` attribute.
    labels : list
        A list of labels for the x-axis.

    Returns
    -------
    None
    """
    boxprops = {"color": "black", "linewidth": 3, "facecolor": "#35478C"}
    medianprops = {"color": "r", "linewidth": 4}
    whiskerprops = {"color": "black", "linewidth": 3}
    capprops = {"color": "black", "linewidth": 3}
    
    errors = [np.array(p.errors) * 100 for p in points_results]
    ax.boxplot(errors, labels=labels, showfliers=False, vert=True, patch_artist=True,
               medianprops=medianprops, boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops)
    ax.tick_params(axis='x', labelrotation=60)
    ax.set_xlabel(r'$\frac{\text{number of clusters}}{\text{number of nodes}}$, $\alpha$')
    ax.set_ylabel('error, %')

def theoretical_max_acceleration_nodes(nodes):
    """
    Calculate the theoretical maximum acceleration for a given number of nodes.

    Parameters
    ----------
    nodes : int
        The number of nodes.

    Returns
    -------
    float
        The theoretical maximum acceleration.
    """
    return 1 / 13 * nodes ** 0.48

def theoretical_max_acceleration_density(density):
    """
    Calculate the theoretical maximum acceleration for a given density.

    Parameters
    ----------
    density : float
        The density.

    Returns
    -------
    float
        The theoretical maximum acceleration.
    """
    return 1 / 13 * (2.8 / density) ** 0.48

def theoretical_alpha_star_nodes(nodes):
    """
    Calculate the theoretical alpha star value for a given number of nodes.

    Parameters
    ----------
    nodes : int
        The number of nodes.

    Returns
    -------
    float
        The theoretical alpha star value.
    """
    return 8.09 * nodes ** -0.48 * (1 - 19.4 / (4.8 * np.log(nodes) + 8.8))

def theoretical_alpha_star_density(density):
    """
    Calculate the theoretical alpha star value for a given density.

    Parameters
    ----------
    density : float
        The density.

    Returns
    -------
    float
        The theoretical alpha star value.
    """
    return 8.09 * (2.8 / density) ** -0.48 * (1 - 19.4 / (4.8 * np.log(2.8 / density) + 8.8))

def plot_loglog(x_data, y_data, theoretical_curve, x_label, y_label, legend_real, legend_theoretical, y_limits=None, x_limits=None):
    """
    Plot a log-log graph with real and theoretical data.

    Parameters
    ----------
    x_data : array-like
        The x-values of the real data.
    y_data : array-like
        The y-values of the real data.
    theoretical_curve : callable
        A function that returns the theoretical y-values for a given set of x-values.
    x_label : str
        The label for the x-axis.
    y_label : str
        The label for the y-axis.
    legend_real : str
        The legend label for the real data.
    legend_theoretical : str
        The legend label for the theoretical data.
    y_limits : tuple, optional
        The limits for the y-axis (default is None).
    x_limits : tuple, optional
        The limits for the x-axis (default is None).
    """

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

def plot_city_results(res, N, max_alpha=1, alpha_threshold=0.5, plot_boxplot=True):
    """
    Plot the results for a city.

    Parameters
    ----------
    res : CityResult
        The results for the city.
    N : int
        The number of nodes.
    max_alpha : float, optional
        The maximum alpha value (default is 1).
    alpha_threshold : float, optional
        The alpha threshold for filtering the results (default is 0.5).
    plot_boxplot : bool, optional
        Whether to plot the boxplot (default is True).
    """
    if plot_boxplot:
        fig, axs = plt.subplots(2, 1)
        fig.set_figwidth(15)
        fig.set_figheight(15)
    else:
        fig, axs = plt.subplots(1, 1)
        fig.set_figwidth(10)
        fig.set_figheight(10)

    alpha = np.array([p.alpha for p in res.points_results])
    speed_up = np.array([p.speed_up[0] for p in res.points_results])
    resolutions = np.array([p.resolution for p in res.points_results])

    # Теоретическое значение alpha*
    a0 = optimal_a_star(N)
    
    # Построение графика ускорения
    plot_acceleration(axs[0], alpha, speed_up, N, a0, theoretical_acceleration, max_alpha)

    for i, (x, y, resolution) in enumerate(zip(alpha, speed_up, resolutions)):
        axs[0].annotate(f'{resolution:.2f}', (x, y), textcoords="offset points", xytext=(0,0), ha='center', fontsize=6)
        axs[0].scatter(x, y, s=30, marker='o')

    if plot_boxplot:
        # Форматирование лейблов для боксплота
        labels = format_labels(res.points_results, alpha_threshold)

        # Построение боксплота ошибок
        plot_boxplot(axs[1], res.points_results, labels)

    fig.suptitle(fr'$N_0$={N}')
    plt.show()


def plot_theoretical_acceleration(N, figsize=(16, 9), max_alpha=1.0):
    """
    Plot the theoretical acceleration curve.

    Parameters
    ----------
    N : int
        The number of nodes.
    max_alpha : float, optional
        The maximum alpha value (default is 1.0).
    """
    alphas = np.arange(0.001, max_alpha, 0.001)  # Используем np.arange вместо range
    fig, ax = plt.subplots(figsize=figsize)  # Используем plt.subplots вместо plt.figure
    acceleration = [theoretical_acceleration(alpha, N) for alpha in alphas]
    a0 = optimal_a_star(N)
    ax.plot(alphas, acceleration, '--')  # Добавляем график
    ax.axvline(x=a0, ymin=0, c='r', linestyle=':', label=rf'$\alpha^*$ = {a0:.3}', linewidth=6)
    # ax.set_xlabel(r'$\alpha$')  # Добавляем метку для оси x
    ax.set_xlabel(r'$\frac{\text{number of clusters}}{\text{number of nodes}}$, $\alpha$')
    ax.set_ylabel(r'Acceleration, $\gamma$')  # Добавляем метку для оси y
    ax.set_title(r'Theoretical Acceleration for $\alpha$')  # Добавляем заголовок
    ax.legend()
    plt.show()  # Показываем график