import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx

from gsber.utils import DataGenerator

def create_boxplot(mistakes: pd.DataFrame) -> plt.Axes:
    """Create a boxplot for the mistakes DataFrame"""
    ax = mistakes.boxplot(showfliers=False, grid=False)
    ax.set_xticks(range(1, len(mistakes.columns) + 1))
    ax.set_xticklabels(mistakes.columns)
    return ax

def add_line_plot(ax: plt.Axes, output: pd.DataFrame) -> plt.Axes:
    """Add a line plot for the output DataFrame to the existing Axes"""
    ax2 = ax.twinx()
    line, = ax2.plot(range(1, len(output['times']) + 1), output['times'], 'ro-', label='Время работы')
    return ax2, line

def add_horizontal_line(ax: plt.Axes, expected_error_line_percent: int) -> plt.Axes:
    """Add a horizontal line to the Axes at the specified percentage"""
    line = ax.axhline(y=expected_error_line_percent, color='g', linestyle='--', label='Желаемая ошибка')
    return line

def configure_axes(ax1: plt.Axes, ax2: plt.Axes) -> None:
    """Configure the Axes labels, titles, and legends"""
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

def save_plot(graph_id: str, save: bool) -> None:
    """Save the plot to a file if save is True"""
    if save:
        directory = "data/img"
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, f"boxplot_{graph_id}.png")
        plt.savefig(file_path, dpi=300)
        print(f"График boxplot был сохранён в {file_path}")

def box_visualisation(
    graph: nx.Graph, 
    output: pd.DataFrame, 
    mistakes: pd.DataFrame, 
    show: bool, 
    save: bool, 
    expected_error_line_percent: int = 10
) -> None:
    """Create a boxplot with a line plot and save it to a file if specified"""
    plt.figure(figsize=(16, 9))
    ax = create_boxplot(mistakes)
    ax2, line = add_line_plot(ax, output)
    add_horizontal_line(ax, expected_error_line_percent)
    configure_axes(ax, ax2)
    if save:
        save_plot(graph.graph['id'], save)
    if show:
        plt.show()

def add_louven_line(H, x_values, y_values) -> plt.Axes:
    """Add a horizontal line to the Axes at the specified value"""
    """Create a plot with the given x and y values"""
    louven_value = DataGenerator.formula_louven(H)
   
    plt.figure(figsize=(16, 9))
    ax = plt.plot(x_values, y_values, 'ro-', label='Асимптотическое время')
    plt.xlabel('k, отношение кластеров к узлам графа')
    plt.ylabel('Эволюция асимптотического времени')
    plt.title('Асимптотическое время работы')

    line = plt.axhline(y=louven_value, color='g', linestyle='--', label='Алгоритм Лувена')
    return line

def save_plot(file_path: str) -> None:
    """Save the plot to a file"""
    directory = "data/img"
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(file_path, dpi=120)
    print(f"График асимптоты был сохранён в {file_path}")

def visualisation(
    graph: nx.Graph, 
    node_counts: List[int], 
    edge_counts: List[int], 
    show: bool, 
    save: bool
) -> None:
    """Create a plot of the asymptotic time and save it to a file if specified"""
    x_values = []
    y_values = []

    for i in range(len(node_counts)):
        k = node_counts[i] / len(graph.nodes())
        y = DataGenerator.formula_centroid(len(graph.nodes()), len(graph.edges()), edge_counts[i], k)
        x_values.append(round(k, 3))
        y_values.append(y)

    # ax = create_plot(x_values, y_values)
    line = add_louven_line(graph, x_values, y_values)

    if save:
        file_path = os.path.join("data", "img", f"асимптота_{graph.graph['id']}.png")
        save_plot(file_path)

    if show:
        plt.show()