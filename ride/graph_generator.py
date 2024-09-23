import itertools
import math
import random
import time

import networkx as nx
import numpy as np
import osmnx as ox
from tqdm import tqdm
from ride.common import GraphLayer
from typing import Union
from ride.utils import extract_cluster_list_subgraph


# def get_dist(du, dv) -> float:
#     d = (du['x'] - dv['x']) ** 2 + (du['y'] - dv['y']) ** 2
#     d = d ** 0.5 / 360 * 2 * np.pi * 6371.01 * 1000
#     return d


# def extract_cluster_subgraph(graph: nx.Graph, cluster_number: int) -> nx.Graph:
#     nodes_to_keep = [node for node, data in graph.nodes(data=True) if data['cluster'] == cluster_number]
#     return graph.subgraph(nodes_to_keep)



# def get_graph(city_id: str = 'R2555133') -> nx.Graph:
#     gdf = ox.geocode_to_gdf(city_id, by_osmid=True)
#     polygon_boundary = gdf.unary_union
#     graph = ox.graph_from_polygon(polygon_boundary,
#                                   network_type='drive',
#                                   simplify=True)
#     G = nx.Graph(graph)
#     H = nx.Graph()
#     # Добавляем рёбра в новый граф, копируя только веса
#     for u, d in G.nodes(data=True):
#         H.add_node(u, x=d['x'], y=d['y'])
#     for u, v, d in G.edges(data=True):
#         H.add_edge(u, v, length=d['length'])
#     del city_id, gdf, polygon_boundary, graph, G
#     return H


def resolve_communities(H: nx.Graph, r: float = 20) -> list[set[int]]:
    """
        Resolve communities in a graph using the Louvain community detection algorithm.

        Parameters
        ----------
        H : nx.Graph
            The graph to resolve communities in.
        r : float, optional
            The resolution parameter for the Louvain algorithm (default is 20).

        Returns
        -------
        list[set[int]]
            A list of sets of node IDs representing the resolved communities.

        Notes
        -----
        This function uses the Louvain community detection algorithm to resolve communities in the graph.
        It also assigns a 'cluster' attribute to each node in the graph with the community ID.
    """
    communities = nx.community.louvain_communities(H,
                                                   seed=1534,
                                                   weight='length',
                                                   resolution=r)
    cls = []
    for i, c in enumerate(communities):
        for n in nx.connected_components(H.subgraph(c)):
            cls.append(n)
    for i, ids in enumerate(cls):
        for j in ids:
            H.nodes[j]['cluster'] = i
    return cls


# def generate_communities_subgraph(H: nx.Graph, communities: list[set[int]]) -> list[nx.Graph]:
#     return [extract_cluster_subgraph(H, i) for i, c in enumerate(communities)]


def get_cluster_to_neighboring_clusters(H: nx.Graph) -> dict[int, set[int]]:
    """
        Returns a dictionary mapping each cluster to its neighboring clusters in the graph.

        Parameters
        ----------
        H : nx.Graph
            The graph to get neighboring clusters from.

        Returns
        -------
        dict[int, set[int]]
            A dictionary where each key is a cluster ID and the value is a set of neighboring cluster IDs.

        Notes
        -----
        This function iterates over the nodes in the graph, and for each node, it finds the neighboring clusters and adds them to the dictionary.
    """
    cls_to_neighboring_cls = {}
    for u in H.nodes():
        du = H.nodes[u]
        for v in H[u]:
            dv = H.nodes[v]
            if dv['cluster'] == du['cluster']:
                continue
            c1 = dv['cluster']
            c2 = du['cluster']
            if not (c1 in cls_to_neighboring_cls):
                cls_to_neighboring_cls[c1] = set()
            if not (c2 in cls_to_neighboring_cls):
                cls_to_neighboring_cls[c2] = set()
            cls_to_neighboring_cls[c1].add(c2)
            cls_to_neighboring_cls[c2].add(c1)
    return cls_to_neighboring_cls


def get_cluster_to_bridge_points(H: nx.Graph) -> dict[int, set[int]]:
    """
        Returns a dictionary mapping each cluster to its bridge points in the graph.

        Parameters
        ----------
        H : nx.Graph
            The graph to get bridge points from.

        Returns
        -------
        dict[int, set[int]]
            A dictionary where each key is a cluster ID and the value is a set of node IDs that are bridge points for that cluster.

        Notes
        -----
        This function iterates over the nodes in the graph, and for each node, it finds the bridge points for each cluster and adds them to the dictionary.
    """
    cls_to_bridge_points = {}
    for u in H.nodes():
        from_node = H.nodes[u]
        for v in H[u]:
            to_node = H.nodes[v]
            if to_node['cluster'] == from_node['cluster']:
                continue
            c1 = to_node['cluster']
            c2 = from_node['cluster']

            if not (c1 in cls_to_bridge_points):
                cls_to_bridge_points[c1] = set()
            if not (c2 in cls_to_bridge_points):
                cls_to_bridge_points[c2] = set()
            cls_to_bridge_points[c2].add(u)
            cls_to_bridge_points[c1].add(v)
    return cls_to_bridge_points


def get_cluster_to_centers(X: nx.Graph) -> dict[int, int]:
    """
        Returns a dictionary mapping each cluster to its center node in the graph.

        Parameters
        ----------
        X : nx.Graph
            The graph to get cluster centers from.

        Returns
        -------
        dict[int, int]
            A dictionary where each key is a cluster ID and the value is the node ID of the cluster center.

        Notes
        -----
        This function creates a dictionary that maps each cluster to its center node in the graph.
    """
    cls_to_center = {d['cluster']: u for u, d in X.nodes(data=True)}
    return cls_to_center


# def get_path_len(d: dict, points, p=1.0):
#     res_len = 0
#     for u in points:
#         if u in d and d[u] > 0:
#             res_len += d[u] ** p
#     if res_len > 0:
#         return res_len ** (1 / p)
#     return 0


def build_center_graph(
        graph: nx.Graph,
        communities: list[set[int]],
        cluster_to_bridge_points: dict[int, set[int]],
        cluster_to_neighboring_cluster: dict[int, set[int]],
        p: float = 1.0,
        use_all_point: bool = True,
        has_coordinates: bool = True) -> nx.Graph:
    """
        Builds a center graph for the given graph G.

        Parameters
        ----------
        graph : nx.Graph
            The original graph.
        communities : list[set[int]]
            A list of communities in the graph.
        cluster_to_bridge_points : dict[int, set[int]]
            A dictionary mapping each cluster to its bridge points.
        cluster_to_neighboring_cluster : dict[int, set[int]]
            A dictionary mapping each cluster to its neighboring clusters.
        p : float, optional
            Not used in this function (default is 1.0).
        use_all_point : bool, optional
            Not used in this function (default is True).
        has_coordinates : bool, optional
            Whether the graph has node coordinates (default is True).

        Returns
        -------
        nx.Graph
            The center graph.

        Notes
        -----
        This function first finds the center node for each community, then adds edges between the center nodes based on the neighboring clusters.
    """
    centers = {}
    X = nx.Graph()
    for cls, _ in enumerate(communities):
        gc = extract_cluster_list_subgraph(graph, [cls], communities)
        min_node = nx.barycenter(gc, weight='length')[0]
        du = graph.nodes(data=True)[min_node]
        X.add_node(min_node, **du)
        centers[cls] = min_node
    if len(X.nodes) == 1:
        return X
    for u, d in X.nodes(data=True):
        for v in cluster_to_neighboring_cluster[d['cluster']]:
            dv = X.nodes[centers[v]]
            if has_coordinates:
                path = np.sqrt((d['x'] - dv['x']) ** 2 + (d['y'] - dv['y']) ** 2)
            else:
                path = nx.single_source_dijkstra(
                    extract_cluster_list_subgraph(graph, [d['cluster'], dv['cluster']], communities),
                    u,
                    centers[v],
                    weight='length'
                )[0]
            X.add_edge(u, centers[v], length=path)
    return X


# def get_node(H: nx.Graph, cls_to_center: dict, X: nx.Graph):
#     node_from = random.choice(list(H.nodes()))
#     node_to = random.choice(list(H.nodes()))
#     path_len = nx.single_source_dijkstra(H, node_from, node_to, weight='length')
#     c = set()
#     for u in path_len[1]:
#         c.add(H.nodes[u]['cluster'])
#     while len(c) < 5:
#         node_from = random.choice(list(H.nodes()))
#         node_to = random.choice(list(H.nodes()))
#         path_len = nx.single_source_dijkstra(H, node_from, node_to, weight='length')
#         c.clear()
#         for u in path_len[1]:
#             c.add(H.nodes[u]['cluster'])
#     return node_from, node_to


def get_node_for_initial_graph(H: nx.Graph):
    """
        Returns two random nodes from the graph H.

        Parameters
        ----------
        H : nx.Graph
            The graph to select nodes from.

        Returns
        -------
        tuple[int, int]
            Two random nodes from the graph.

        Notes
        -----
        This function selects two random nodes from the graph, ensuring they are not the same.
    """
    nodes = list(H.nodes())
    f, t = random.choice(nodes), random.choice(nodes)
    while f == t:
        f, t = random.choice(nodes), random.choice(nodes)
    return f, t


def generate_layer(H: nx.Graph, resolution: Union[float, int], p: float = 1, use_all_point: bool = True, communities=None,
                   has_coordinates: bool = False, times=False) -> GraphLayer:
    """
        Generates a graph layer based on the input graph and parameters.

        Parameters
        ----------
        H : nx.Graph
            The input graph.
        resolution : float
            The resolution for community detection.
        p : float, optional
            Not used in this function (default is 1).
        use_all_point : bool, optional
            Not used in this function (default is True).
        communities : list[set[int]], optional
            Pre-computed communities (default is None).
        has_coordinates : bool, optional
            Whether the graph has coordinates (default is False).

        Returns
        -------
        GraphLayer
            The generated graph layer.

        Notes
        -----
        This function first resolves the communities in the graph, then builds the neighboring clusters, bridge points, and centroid graph.
    """

    start = time.time()
    if communities is None:
        communities = resolve_communities(H, resolution)
    build_communities = time.time() - start
    start = time.time()
    cluster_to_neighboring_clusters = get_cluster_to_neighboring_clusters(H)
    cluster_to_bridge_points = get_cluster_to_bridge_points(H)
    build_additional = time.time() - start
    start = time.time()

    centroids_graph = build_center_graph(
        graph=H,
        communities=communities,
        cluster_to_bridge_points=cluster_to_bridge_points,
        cluster_to_neighboring_cluster=cluster_to_neighboring_clusters,
        p=p,
        use_all_point=use_all_point,
        has_coordinates=has_coordinates
    )
    build_centroid_graph = time.time() - start

    cluster_to_centers = get_cluster_to_centers(centroids_graph)

    layer: GraphLayer = GraphLayer(
        H,
        communities,
        cluster_to_neighboring_clusters,
        cluster_to_bridge_points,
        cluster_to_centers,
        centroids_graph
    )
    if times:
        return layer, build_communities, build_additional, build_centroid_graph
    return layer