import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx

from ride.utils import DataGenerator

def create_boxplot(errors: pd.DataFrame) -> plt.Axes:
    """
    Create a boxplot from a DataFrame containing errors.

    Parameters
    ----------
    errors : pd.DataFrame
        A DataFrame containing errors.

    Returns
    -------
    plt.Axes
        The axis with the boxplot.
    """
    
    ax = errors.boxplot(showfliers=False, grid=False)
    ax.set_xticks(range(1, len(errors.columns) + 1))
    ax.set_xticklabels(errors.columns)
    return ax

def _add_line_plot(ax: plt.Axes, times: list) -> plt.Axes:
    """
    Add a line plot representing the execution time to an existing boxplot.

    Parameters
    ----------
    ax : plt.Axes
        The axis with the boxplot.
    times : list
        A list of execution times.

    Returns
    -------
    plt.Axes
        The new axis with the line plot.
    """

    ax2 = ax.twinx()
    line, = ax2.plot(range(1, len(times) + 1), times, 'ro-', label='Время работы')
    return ax2, line

def _add_horizontal_line(ax: plt.Axes, expected_error_line_percent: int, color:str='g') -> plt.Axes:
    """
    Add a horizontal line to a plot representing the desired error level.

    Parameters
    ----------
    ax : plt.Axes
        The axis to add the line to.
    expected_error_line_percent : int
        The desired error level.
    color : str, optional
        The color of the line (default is 'g').

    Returns
    -------
    plt.Axes
        The axis with the horizontal line.
    """

    line = ax.axhline(y=expected_error_line_percent, color=color, linestyle='--', label='Желаемая ошибка')
    return line

def _configure_axes(ax1: plt.Axes, ax2: plt.Axes) -> None:
    """
    Configure the labels, titles, and legends of two axes.

    Parameters
    ----------
    ax1 : plt.Axes
        The first axis.
    ax2 : plt.Axes
        The second axis.
    """

    ax1.set_ylabel('Ошибки, %', color='b')
    ax2.set_ylabel('Время, сек', color='r')
    for tl in ax1.get_yticklabels():
        tl.set_color('b')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')
    ax1.set_xlabel('k, отношение кластеров к узлам графа')
    plt.title('Поиск оптимального отношения кластеров к кол-ву узлов в графе')
    lines = [line for line in [ax1, ax2] if line]
    ax1.legend(lines, [l.get_label() for l in lines])


def box_visualisation(
    graph: nx.Graph, 
    times: list, 
    dijkstra_time: float,
    errors: pd.DataFrame, 
    show: bool = True, 
    save: bool = False, 
    expected_error_line_percent: int = 10
) -> None:
    """
    Create a boxplot with a line plot and save it to a file if specified.

    Parameters
    ----------
    graph_id : str
        The ID of the graph.
    errors : pd.DataFrame
        A DataFrame containing errors.
    times : list
        A list of execution times.
    dijkstra_time : float
        The execution time of Dijkstra's algorithm.
    show : bool, optional
        Whether to show the plot (default is True).
    save : bool, optional
        Whether to save the plot (default is False).
    expected_error_line_percent : int, optional
        The desired error level (default is 10).
    """

    plt.figure(figsize=(16, 9))
    ax = create_boxplot(errors)
    ax2, line = _add_line_plot(ax, times)
    _add_horizontal_line(ax, expected_error_line_percent)
    _add_horizontal_line(ax2, dijkstra_time, color='r')

    _configure_axes(ax, ax2)
    if save:
        save_plot(graph.graph['id'])
    if show:
        plt.show()

def add_louven_line(H, x_values, y_values) -> plt.Axes:
    """
    Add a horizontal line to a plot representing the Louvain algorithm.

    Parameters
    ----------
    H : 
        The graph.
    x_values : 
        The x-values of the plot.
    y_values : 
        The y-values of the plot.

    Returns
    -------
    plt.Axes
        The axis with the horizontal line.
    """

    louven_value = DataGenerator.formula_louven(H)
   
    plt.figure(figsize=(16, 9))
    ax = plt.plot(x_values, y_values, 'ro-', label='Асимптотическое время')
    plt.xlabel('k, отношение кластеров к узлам графа')
    plt.ylabel('Эволюция асимптотического времени')
    plt.title('Асимптотическое время работы')
    # plt.legend(['Расчетное время на всём графе', 'расчетное время на центроидах по формуле'])

    line = plt.axhline(y=louven_value, color='g', linestyle='--', label='Алгоритм Лувена')
    return line


def save_plot(graph_id: str, save: bool) -> None:
    """
    Save a plot to a file if specified.

    Parameters
    ----------
    graph_id : str
        The ID of the graph.
    save : bool
        Whether to save the plot.
    """

    if save:
        directory = "data/img"
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, f"boxplot_{graph_id}.png")
        plt.savefig(file_path, dpi=300)
        print(f"График boxplot был сохранён в {file_path}")


def visualisation(
    graph: nx.Graph, 
    node_counts: List[int], 
    edge_counts: List[int], 
    show: bool, 
    save: bool
) -> None:
    """
    Create a plot of the asymptotic time and save it to a file if specified.

    Parameters
    ----------
    graph : nx.Graph
        The graph.
    node_counts : List[int]
        A list of node counts.
    edge_counts : List[int]
        A list of edge counts.
    show : bool
        Whether to show the plot.
    save : bool
        Whether to save the plot.
    """
    x_values = []
    y_values = []

    for i in range(len(node_counts)):
        k = node_counts[i] / len(graph.nodes())
        y = DataGenerator.formula_centroid(len(graph.nodes()), len(graph.edges()), edge_counts[i], k)
        x_values.append(round(k, 3))
        y_values.append(y)

    # ax = create_plot(x_values, y_values)
    _ = add_louven_line(graph, x_values, y_values)

    if save:
        file_path = os.path.join("data", "img", f"асимптота_{graph.graph['id']}.png")
        save_plot(file_path)

    if show:
        plt.show()