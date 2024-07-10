import os
import pandas as pd
import networkx as nx
from tqdm import tqdm
from multiprocessing import Pool, Manager

def louvain_clusters(H, seed=42, weight='length', resolution=1):
    communities = nx.community.louvain_communities(H, seed=seed,
                                                weight=weight,
                                                resolution=resolution)
    return H, communities


# Функция для поиска разрешений
def search_resolutions(H, resolution=0.001, weight='length', k_max=0.7):
    resolutions = []
    k = 0

    while k < k_max:
      H, communities = louvain_clusters(H, resolution=resolution, weight=weight)
      k = len(communities)/len(H.nodes)
      if k < 0.008:
        resolution *= 3
        continue
      else:
        resolutions.append(resolution)
        resolution *= 3

    return resolutions

# Функция обработки графа
def process_graph(args):
    number, return_dict = args
    print(f"Processing graph {number}")
    try:
        H = nx.read_graphml(f'data/graphs_1/graph_R{number}.graphml')
    except FileNotFoundError:
        print(f"File for R{number} not found.")
        return

    resolutions = search_resolutions(H, k_max=0.7)

    for resolution in resolutions:
        communities = nx.community.louvain_communities(H, seed=42, weight='length', resolution=resolution)
        alpha = len(communities) / len(H.nodes)
        modularity = nx.community.modularity(H, communities)

        if number not in return_dict:
            return_dict[number] = {}
        return_dict[number][round(alpha, 3)] = modularity

# Функция для поиска ближайшего значения
def find_closest_value(alpha, alpha_dict):
    return min(alpha_dict.keys(), key=lambda x: abs(x - alpha))

# Функция для получения модулярности
def get_modularity(row, diction):
    r_number = row['R_number']
    alpha = round(row['alpha'], 3)
    if r_number in diction:
        closest_alpha = find_closest_value(alpha, diction[r_number])
        return diction[r_number][closest_alpha]
    else:
        return None

if __name__ == "__main__":
    # Загрузка данных
    df = pd.read_csv('more_info_for_cities_3.csv')
    result = df.loc[df.groupby('R_number')['speed_up'].idxmax()]
    numbers = list(result['R_number'].values)

    manager = Manager()
    return_dict = manager.dict()

    # Узнаем количество доступных ядер
    num_cores = os.cpu_count()
    print(f"Available CPU cores: {num_cores}")

    # Задаем количество потоков для пула
    num_threads = num_cores

    with Pool(processes=num_threads) as pool:
        list(tqdm(pool.imap(process_graph, [(number, return_dict) for number in numbers]), total=len(numbers), desc="Processing graphs"))

    diction = dict(return_dict)

    df['modularity'] = df.apply(lambda row: get_modularity(row, diction), axis=1)

    # Сохранение DataFrame в CSV файл
    df.to_csv('output_dataframe.csv', index=False)