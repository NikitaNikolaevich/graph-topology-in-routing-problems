import math
import pickle
import random
import sys
from multiprocessing import Pool

import networkx as nx
import numpy as np
from scipy.spatial import KDTree as kd
from tqdm import trange, tqdm

import city_tests
import graph_generator

import osmnx as ox


def get_graph(city_id: str = 'R2555133') -> nx.Graph:
    gdf = ox.geocode_to_gdf(city_id, by_osmid=True)
    polygon_boundary = gdf.unary_union
    graph = ox.graph_from_polygon(polygon_boundary,
                                  network_type='drive',
                                  simplify=True)
    G = nx.Graph(graph)
    H = nx.Graph()
    # Добавляем рёбра в новый граф, копируя только веса
    for u, d in G.nodes(data=True):
        H.add_node(u, x=d['x'], y=d['y'])
    for u, v, d in G.edges(data=True):
        H.add_edge(u, v, length=d['length'])
    del city_id, gdf, polygon_boundary, graph, G
    return H


def calculate(data):
    cities = data[0]
    points_number = data[1]
    NUMBER = data[2]
    THREADS = data[3]

    for name, id in cities:
        G = get_graph(id)

        points = [graph_generator.get_node_for_initial_graph_v2(G) for _ in
                  range(points_number)]

        for u in G.nodes:
            if u in G[u]:
                G.remove_edge(u, u)
        # city_tests.test_graph(G,
        #                       f'{name}_dijkstra',
        #                       id,
        #                       points=points, pos=NUMBER, logs=True)
        # city_tests.test_graph(G,
        #                       f'{name}_bidirectional',
        #                       id,
        #                       points=points, pos=NUMBER, logs=True, alg='bidirectional')
        city_tests.test_graph(G,
                              f'{name}_astar',
                              id,
                              points=points, pos=NUMBER, logs=True, alg='astar')
        # print(name, id)
        NUMBER += THREADS


if __name__ == '__main__':
    total = 1
    points_number = 500
    if len(sys.argv) == 2:
        total = int(sys.argv[1])

    print('THREADS:', total)
    print('POINTS:', points_number)

    cities = {
        # 'ASHA': 'R13470549',
        # 'KRG': 'R4676636',
        # 'EKB': 'R6564910',
        # 'BARCELONA': 'R347950',
        'PARIS': 'R71525',
        # 'Prague': 'R435514',
        # 'MSK': 'R2555133',
        # 'SBP': 'R337422',
        # 'SINGAPORE': 'R17140517',
        # 'BERLIN': 'R62422',
        # 'ROME': 'R41485',
        # 'LA': 'R207359',
        # 'DUBAI': 'R4479752',
        # 'RIO': 'R2697338',
        # 'DELHI': 'R1942586',
        # 'KAIR': 'R5466227'
    }
    # for c in cities:
    #     if os.fi
    # G = get_graph('R71525')
    # nx.write_graphml(G, f'{cities["ASHA"]}')
    # with open('PARIS.pkl', 'wb') as fp:
    #     pickle.dump(G, fp)
    #     fp.close()
    total_len = len(cities)
    l = list(cities.items())
    data = [[l[i: total_len: total], points_number, i + 1, total] for i in range(total)]
    # print(data)
    with Pool(total) as p:
        p.map(calculate, data)
