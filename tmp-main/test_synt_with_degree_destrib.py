import math
import os
import pickle
import random
import sys
from multiprocessing import Pool

import networkx as nx
import numpy as np
from scipy.spatial import KDTree as kd
from tqdm import tqdm, trange

from tests import city_tests
from ride import graph_generator
from tqdm.contrib.concurrent import process_map


def gen(N):
    W, H = 1000, 1000
    Q = nx.Graph()
    for i in range(N):
        x = random.random() * W
        y = random.random() * H
        Q.add_node(i, x=x, y=y)
    return Q


def add_density(H: nx.Graph, neighbour: dict[int, int]) -> nx.Graph:
    _G = H.copy()
    ids = [node for node in H.nodes()]
    points = [[d['x'], d['y']] for u, d in H.nodes(data=True)]

    tree = kd(points)
    # random.seed(123)
    for u, du in H.nodes(data=True):
        exist = len(_G[u])
        if exist >= neighbour[u]:
            continue
        dists, n_ids = tree.query([du['x'], du['y']], 300)
        if type(n_ids) is np.int64:
            n_ids = [n_ids]
            dists = [dists]
        total = len(n_ids)
        count = 0
        for i in range(total):
            _id = n_ids[i]
            d = dists[i]
            if ids[_id] == u or len(_G[ids[_id]]) >= neighbour[ids[_id]]:
                continue
            if count >= neighbour[u]:
                break
            count += 1
            _G.add_edge(u, ids[_id], length=d)
        # if count!=neighbour[u]:
        #     print(count, neighbour[u])
    if not nx.is_connected(_G):
        print('fix connected')
        print(len(list(nx.connected_components(_G))))
        tmp = []
        # components = [n for n in nx.connected_components(_G)]
        # distance = {}
        # for i in trange(len(components), pos = 2):
        #     for j in range(i + 1, len(components)):
        #         for u in components[i]:
        #             du = _G.nodes[u]
        #             for v in components[j]:
        #                 dv = _G.nodes[v]
        #                 dst = ((du['x'] - dv['x']) ** 2 + (du['y'] - dv['y']) ** 2) ** 0.5
        #                 if i not in distance:
        #                     distance[i] = {}
        #                 if j not in distance[j]:
        #                     distance[i][j] = (dst, u, v)
        #                 elif distance[i][j][0] > dst:
        #                     distance[i][j] = (dst, u, v)
        for n in nx.connected_components(_G):
            for q in n:
                tmp.append(q)
                break
        for e in _G.edges:
            de = _G.edges[e]
            if 'length' not in de:
                print(1)
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
    points_number = data[1]
    NUMBER = data[2]
    THREADS = data[3]
    ii = data[4]
    W = data[5]

    degrees = np.random.choice(range(0, len(W)), size=N, p=W)

    G = gen(N)
    Q = add_density(G, degrees)
    while not nx.is_connected(Q):
        tqdm.write('shuffle ' + str(len(list(nx.connected_components(Q)))))
        random.shuffle(degrees)
        G = gen(N)
        Q = add_density(G, degrees)
    G = Q

    tqdm.write(str(len(G.edges) / N))

    points = [graph_generator.get_node_for_initial_graph_v2(G) for _ in
              range(points_number)]

    for u in Q.nodes:
        if u in Q[u]:
            Q.remove_edge(u, u)
    for i in range(1):
        city_tests.test_graph(Q,
                              f'PlanePoints{ii}_{len(G.nodes)}_{round(nx.density(Q) * 10000) / 10000}',
                              '0',
                              points=points, pos=NUMBER)


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
    elif len(sys.argv) == 3:
        total = int(sys.argv[1])
        points_number = int(sys.argv[2])

    print('THREADS:', total)
    print('POINTS:', points_number)

    degree_destrib = {
        0: 0,
        3: 0.816342,
        2: 0.044013,
        1: 0.071698,
        4: 0.061802,
        5: 0.005618,
        7: 0.000017
    }

    q = list(degree_destrib.items())
    q.sort(key=lambda x: x[0])
    degree_destrib = dict(q)
    W = list(degree_destrib.values())

    W = np.array(W)
    W /= np.sum(W)

    NODES = [i for i in range(1000, 50000, 5000)]
    total_len = len(NODES)
    data = [[N, points_number, j + 1, total, 0, W] for j, N in enumerate(NODES)]
    data.sort(key=lambda x: x[0])
    with Pool(total) as p:
        res = list(tqdm(p.imap(calculate, data), total=len(data), position=0, desc='tests graphs'))
