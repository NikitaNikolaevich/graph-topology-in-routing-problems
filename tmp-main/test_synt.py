import math
import os
import pickle
import random
import sys
from multiprocessing import Pool

import networkx as nx
import numpy as np
from scipy.spatial import KDTree as kd

from tests import city_tests
from ride import graph_generator


def gen(N):
    W, H = 1000, 1000
    Q = nx.Graph()
    for i in range(N):
        x = random.random() * W
        y = random.random() * H
        Q.add_node(i, x=x, y=y)
    return Q


def add_density(H: nx.Graph, r) -> nx.Graph:
    _G = H.copy()
    ids = [node for node in H.nodes()]
    points = [[d['x'], d['y']] for u, d in H.nodes(data=True)]

    tree = kd(points)
    random.seed(123)
    prob = r - int(r)
    for u, du in H.nodes(data=True):

        dists, n_ids = tree.query([du['x'], du['y']], math.ceil(r))
        if type(n_ids) is np.int64:
            n_ids = [n_ids]
            dists = [dists]
        if math.ceil(r) == 1:
            total = len(n_ids)
        else:
            total = len(n_ids) - 1
            if random.random() < prob:
                total += 1
        for i in range(total):
            _id = n_ids[i]
            d = dists[i]
            if ids[_id] == u:
                continue
            _G.add_edge(u, ids[_id], length=d)
    if not nx.is_connected(_G):
        # print('fix connected')
        tmp = []
        for n in nx.connected_components(_G):
            for q in n:
                tmp.append(q)
                break
        for i in range(len(tmp) - 1):
            d1 = _G.nodes[tmp[i]]
            d2 = _G.nodes[tmp[i + 1]]
            _G.add_edge(tmp[i], tmp[i + 1], length=((d1['x'] - d2['x']) ** 2 + (d1['y'] - d2['y']) ** 2) ** 0.5)
    return _G


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
    return G


def calculate(data):
    N = data[0]
    dens = data[1]
    points_number = data[2]
    NUMBER = data[3]
    THREADS = data[4]
    ii = data[5]
    with open(f'{N}.pkl', 'rb') as file:
        G = pickle.load(file)
        file.close()

    points = [graph_generator.get_node_for_initial_graph_v2(G) for _ in
              range(points_number)]

    for d in dens:
        k = d * (N - 1)
        Q = add_density(G, k)
        for u in Q.nodes:
            if u in Q[u]:
                Q.remove_edge(u, u)
        for i in range(1):
            city_tests.test_graph(Q,
                                  f'PlanePoints{ii}_{len(G.nodes)}_{round(nx.density(Q) * 10000) / 10000}',
                                  '0',
                                  points=points, pos=NUMBER)

        NUMBER += THREADS


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

        k = d * (N - 1)
        Q = G
        for u in Q.nodes:
            if u in Q[u]:
                Q.remove_edge(u, u)
        for i in range(1):
            city_tests.test_graph(Q,
                                  f'Rand{ii}_{len(G.nodes)}_{round(nx.density(Q) * 10000) / 10000}',
                                  '0',
                                  points=points, pos=NUMBER)
        NUMBER += THREADS


if __name__ == '__main__':
    total = 1
    points_number = 500
    if len(sys.argv) == 2:
        total = int(sys.argv[1])
    else:
        total = int(sys.argv[1])
        points_number = int(sys.argv[2])

    print('THREADS:', total)
    print('POINTS:', points_number)

    dens = [0.0022981490745372685]
    while dens[-1] * 1.6 < 1:
        dens.append(dens[-1] * 1.3)
    dens.append(1)
    dens = np.array(dens)
    # dens = dens[dens < 0.05]

    for ii in range():
        NODES = [2000]
        for N in NODES:
            if os.path.isfile(f'{N}.pkl'):
                continue
            G = gen(N)
            with open(f'{N}.pkl', 'wb') as fp:
                pickle.dump(G, fp)
                fp.close()

        total_len = len(dens)
        data = [[N, dens[i: total_len: total], points_number, j * total_len + (i + 1), total,ii] for i in range(total) for
                j, N in enumerate(NODES)]
        data.sort(key=lambda x: x[0])
        with Pool(total) as p:
            p.map(calculate, data)
        os.remove(f'{N}.pkl')

    # for ii in range(5):
    #     NODES = [2000]
    #     total_len = len(dens)
    #     data = [[N, dens[i: total_len: total], points_number, j * total_len + (i + 1), total,ii] for i in range(total) for
    #             j, N in enumerate(NODES)]
    #     data.sort(key=lambda x: x[0])
    #     with Pool(total) as p:
    #         p.map(calculate_rand, data)
