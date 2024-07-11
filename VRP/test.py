import os
import pandas as pd
import networkx as nx
from multiprocessing import Pool

def louvain_clusters(H, seed=42, weight='length', resolution=1):
    communities = nx.community.louvain_communities(H, seed=seed, weight=weight, resolution=resolution)
    return H, communities

def search_resolutions(H, resolution=0.001, weight='length', k_max=0.7):
    resolutions = []
    k = 0

    while k < k_max:
        H, communities = louvain_clusters(H, resolution=resolution, weight=weight)
        k = len(communities) / len(H.nodes)
        if k < 0.008:
            resolution *= 3
            continue
        else:
            resolutions.append(resolution)
            resolution *= 3

    return resolutions

def process_graph(number):
    return_dict = {}
    try:
        H = nx.read_graphml(f'data/graphs_1/graph_R{number}.graphml')
    except FileNotFoundError:
        print(f"File for R{number} not found.")
        return return_dict

    resolutions = search_resolutions(H, k_max=0.7)

    for resolution in resolutions:
        communities = nx.community.louvain_communities(H, seed=42, weight='length', resolution=resolution)
        alpha = len(communities) / len(H.nodes)
        modularity = nx.community.modularity(H, communities)

        if number not in return_dict:
            return_dict[number] = {}
        return_dict[number][round(alpha, 3)] = modularity

    print(f"Processed graph R{number}")

    return return_dict

def find_closest_value(alpha, alpha_dict):
    return min(alpha_dict.keys(), key=lambda x: abs(x - alpha))

def get_modularity(row, diction):
    r_number = row['R_number']
    alpha = round(row['alpha'], 3)
    if r_number in diction:
        closest_alpha = find_closest_value(alpha, diction[r_number])
        return diction[r_number][closest_alpha]
    else:
        return None

if __name__ == "__main__":
    df = pd.read_csv('more_info_for_cities_3.csv')
    result = df.loc[df.groupby('R_number')['speed_up'].idxmax()]
    numbers = list(result['R_number'].values)

    # Создание пула процессов с использованием всех доступных CPU ядер
    with Pool(processes=os.cpu_count()) as pool:
        results = pool.map(process_graph, numbers)

    return_dict = {}
    for result in results:
        for key, value in result.items():
            return_dict[key] = value

    df['modularity'] = df.apply(lambda row: get_modularity(row, return_dict), axis=1)

    df.to_csv('output_dataframe_test_py.csv', index=False)

    print("Processing completed.")
