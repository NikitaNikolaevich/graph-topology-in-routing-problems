import math
import os
import pickle
import random
import sys
from multiprocessing import Pool

import networkx as nx
import numpy as np
from scipy.spatial import KDTree as kd

import city_tests
import graph_generator


def get_rand_graph(N, p):
    G = nx.fast_gnp_random_graph(N, p, directed=False)
    if not nx.is_connected(G):
        tmp = []
        for n in nx.connected_components(G):
            for q in n:
                tmp.append(q)
                break
        for i in range(len(tmp) - 1):
            G.add_edge(tmp[i], tmp[i + 1])
    for e in G.edges:
        G.add_edge(e[0], e[1], length=np.random.random_sample() + 0.001)
    for u in G.nodes:
        if u in G[u]:
            G.remove_edge(u, u)
    return G


def calculate_rand(data):
    N = data[0]
    dens = data[1]
    points_number = data[2]
    NUMBER = data[3]
    THREADS = data[4]
    ii = data[5]
    for d in dens:
        G = get_rand_graph(N, max(min(1, d), 0))

        points = [graph_generator.get_node_for_initial_graph_v2(G) for _ in
                  range(points_number)]
        r = city_tests.get_resolution_for_alpha(G, 0.2)
        Q = G

        for i in range(1):
            city_tests.test_graph(Q,
                                  f'Rand{ii}_{len(G.nodes)}_{round(nx.density(Q) * 10000) / 10000}',
                                  '0',
                                  points=points, pos=NUMBER, resolutions=[r])
        NUMBER += THREADS


if __name__ == '__main__':
    total = 1
    points_number = 500
    if len(sys.argv) == 2:
        total = int(sys.argv[1])
    elif len(sys.argv) == 3:
        total = int(sys.argv[1])
        points_number = int(sys.argv[2])

    print('THREADS:', total)
    print('POINTS:', points_number)

    dens = [0.0022981490745372685]
    while dens[-1] * 1.6 < 1:
        dens.append(dens[-1] * 1.3)
    dens.append(1)
    dens.append(0.6)
    dens.append(0.8)
    dens = np.array(dens)
    # dens = dens[dens < 0.05]

    for ii in range(1):
        NODES = [2000]
        total_len = len(dens)
        data = [[N, dens[i: total_len: total], points_number, j * total_len + (i + 1), total, ii] for i in range(total)
                for
                j, N in enumerate(NODES)]
        data.sort(key=lambda x: x[0])
        with Pool(total) as p:
            p.map(calculate_rand, data)
