import networkx as nx
import pandas as pd
import numpy as np
import random
import math
import time
from tqdm import tqdm
from typing import Dict, Tuple, List
from gsber.utils import DataGenerator

class GraphProcessor:
    def create_G_centroid(H: nx.Graph) -> Tuple[nx.Graph, Dict[int, List[int]]]:
        """Create a graph with centroids of clusters"""
        clusters = GraphProcessor._get_clusters(H)
        cluster_transitions = GraphProcessor._get_cluster_transitions(clusters, H)
        centroids = GraphProcessor._get_centroids(clusters, H)
        G = GraphProcessor._create_graph_from_dict(H, cluster_transitions, centroids)
        return G, clusters

    def louvain_clusters(H: nx.Graph, seed: int = 0, weight: str = 'length', resolution: float = 1) -> Tuple[nx.Graph, Dict[int, List[int]]]:
        """Apply Louvain clustering algorithm to the graph"""
        communities = nx.community.louvain_communities(H, seed=seed, weight=weight, resolution=resolution)
        GraphProcessor._add_clusters_to_nodes(H, communities)
        return H, communities

    def create_points_for_test(H: nx.Graph, min_distance: int = 10) -> pd.DataFrame:
        """Create points for test"""
        H, _ = GraphProcessor.louvain_clusters(H, resolution=1, weight='length')
        G_centroid, clusters = GraphProcessor.create_G_centroid(H)
        random.seed(1)
        res = DataGenerator.sample_nodes_for_experiment(G_centroid, clusters, min_distance)
        return res

    def search_resolutions(H: nx.Graph, resolution: float = 0.001, weight: str = 'length', k_max: float = 0.7) -> Tuple[List[float], List[float], List[int], List[int]]:
        """Search resolutions for Louvain algorithm"""
        resolutions = []
        k = 0
        ks = []
        v1 = []
        e1 = []

        while k < k_max:
            H, communities = GraphProcessor.louvain_clusters(H, resolution=resolution, weight=weight)
            k = len(communities)/len(H.nodes)
            if k < 0.008:
                resolution *= 3
                continue
            else:
                G_centroid, _ = GraphProcessor.create_G_centroid(H)
                if len(G_centroid.nodes()) > 0 and len(G_centroid.edges()) > 0:
                    v1.append(len(G_centroid.nodes()))
                    e1.append(len(G_centroid.edges()))
                    resolutions.append(resolution)
                    ks.append(k)
                    resolution *= 3

        return resolutions, ks, v1, e1

    def _get_clusters(H: nx.Graph) -> Dict[int, List[int]]:
        """Get clusters from the graph"""
        clusters = {}
        for node, data in H.nodes(data=True):
            cluster = data['cluster']
            if cluster not in clusters:
                clusters[cluster] = []
            clusters[cluster].append(node)
        return clusters

    def _get_cluster_transitions(clusters: Dict[int, List[int]], H: nx.Graph) -> Dict[int, List[int]]:
        """Get cluster transitions from the graph"""
        cluster_transitions = {}
        for cluster, nodes in clusters.items():
            neighboring_clusters = set()
            for node in nodes:
                for neighbor in H.neighbors(node):
                    neighbor_cluster = H.nodes[neighbor]['cluster']
                    if neighbor_cluster!= cluster:
                        neighboring_clusters.add(neighbor_cluster)
            cluster_transitions[cluster] = list(neighboring_clusters)
        return cluster_transitions

    def _get_centroids(clusters: Dict[int, List[int]], H: nx.Graph) -> Dict[int, int]:
        """Get centroids from the graph"""
        centroids = {}
        for i in range(len(clusters)):
            nodes_ = [node for node, data in H.nodes(data=True) if data.get('cluster') == i]
            s = H.subgraph(nodes_)
            closeness_centrality = nx.closeness_centrality(s)
            centroid_node = max(closeness_centrality, key=closeness_centrality.get)
            centroids[i] = centroid_node
        return centroids
    
    def _create_graph_from_dict(H, cluster_transitions: Dict[int, List[int]], centroids: Dict[int, int]) -> nx.Graph:
        """Create a graph from a dictionary"""
        G = nx.Graph()
        for node, neighbors in cluster_transitions.items():
            for neighbor in neighbors:
                nodes_ = [n for n, data in H.nodes(data=True) if data.get('cluster') in (node, neighbor)]
                s = H.subgraph(nodes_)
                length, path = nx.single_source_dijkstra(H, centroids[node], centroids[neighbor], weight='length')
                G.add_edge(node, neighbor, weight=length)
        return G

    def _add_clusters_to_nodes(H: nx.Graph, communities: Dict[int, List[int]]) -> None:
        """Add clusters to nodes in the graph"""
        for i, ids in enumerate(communities):
            for j in ids:
                H.nodes[j]['cluster'] = i

    
class GraphRunner:
    def test(H: nx.Graph, resolutions: list, k: int=100, min_distance: int=10, weight:str='length'):
        """
        Test the performance of the graph clustering algorithm on a given graph.

        Parameters:
        H (nx.Graph): The input graph.
        resolutions (list): A list of resolution values to use for the Louvain clustering algorithm.
        k (int): The number of random pairs of nodes to test.
        min_distance (int): The minimum distance between the nodes in the random pairs.
        weight (str): The name of the edge attribute to use as the weight.

        Returns:
        A pandas DataFrame containing the results of the test.
        """

        # Create a list of random pairs of nodes
        res = GraphProcessor.create_points_for_test(H, min_distance=min_distance)

        try:
            part = random.sample(res, k=k)
        except ValueError:
            part = res

        output = {
            'error': [],
            'time_centr': [],
            'times': [],
            'ks': [],
        }

        # Compute the shortest path length between the pairs of nodes using Dijkstra's algorithm
        output, all_length = GraphRunner.compute_shortest_path_length_dijkstra(H, part, weight, output)

        # Compute the shortest path length between the pairs of nodes using the centroid graph
        output, mistakes = GraphRunner.compute_shortest_path_length_centroid(H, resolutions, part, all_length, weight, output)

        output = pd.DataFrame(output)
        mistakes = pd.DataFrame(mistakes)

        return output, mistakes

    def compute_shortest_path_length_dijkstra(H: nx.Graph, part: list, weight: str, output):
        """
        Compute the shortest path length between the pairs of nodes using Dijkstra's algorithm.

        Parameters:
        H (nx.Graph): The input graph.
        part (list): A list of random pairs of nodes.
        weight (str): The name of the edge attribute to use as the weight.

        Returns:
        A list of shortest path lengths between the pairs of nodes.
        """

        start = time.time()
        all_length = []
        print('\nТестирование на начальном графе:')
        for i in tqdm(part):
            length, path1 = nx.single_source_dijkstra(H, i[0], i[1], weight=weight)
            all_length.append(length)
        end = time.time()
        output['dijkstra_time'] = end - start

        return output, all_length

    def compute_shortest_path_length_centroid(H: nx.Graph, resolutions: list, part: list, all_length: list, weight: str, output):
        """
        Compute the shortest path length between the pairs of nodes using the centroid graph.

        Parameters:
        H (nx.Graph): The input graph.
        resolutions (list): A list of resolution values to use for the Louvain clustering algorithm.
        part (list): A list of random pairs of nodes.
        all_length (list): A list of shortest path lengths between the pairs of nodes computed using Dijkstra's algorithm.
        weight (str): The name of the edge attribute to use as the weight.

        Returns:
        A pandas DataFrame containing the results of the test.
        """
        mistakes = {
        
        }
        v1 = []
        e1  =[]
        print('\nТестирование на центроидах:')
        for resolution in resolutions:
            H, communities = GraphProcessor.louvain_clusters(H, resolution=resolution, weight=weight)
            k = len(communities)/len(H.nodes)
            output['ks'].append(round(k, 3))

            start_centr = time.time()
            G_centroid, clusters = GraphProcessor.create_G_centroid(H)
            end_centr = time.time() - start_centr
            output['time_centr'].append(end_centr)
            v1.append(len(G_centroid.nodes()))
            e1.append(len(G_centroid.edges()))

            all_l_c = []
            start = time.time()
            for i in tqdm(part):

                cluster_1 = H.nodes(data=True)[i[0]]['cluster']
                cluster_2 = H.nodes(data=True)[i[1]]['cluster']
                leng_G = nx.dijkstra_path(G_centroid, cluster_1, cluster_2)
                nodes = []
                for g_name in leng_G:
                    nodes.extend(clusters[g_name])
                Hs = H.subgraph(nodes)
                length,path2 = nx.single_source_dijkstra(Hs, i[0], i[1], weight='length')

                all_l_c.append(length)

            result = time.time() - start
            output['times'].append(result)
            mis = (np.array(all_l_c).sum() - np.array(all_length).sum()) / np.array(all_length).sum() * 100
            mis_box_plot = (np.array(all_l_c) - np.array(all_length)) / np.array(all_length) * 100
            mistakes[round(k, 3)] = mis_box_plot
            output['error'].append(mis)

        return output, mistakes
    
if __name__ == "__main__":
    # Example usage of the GraphProcessor class
    graph = nx.erdos_renyi_graph(100, 0.05)
    gp = GraphProcessor()
    G, clusters = gp.create_G_centroid(graph)
    print("Number of clusters:", len(clusters))
    print("Number of nodes in G:", G.number_of_nodes())
    print("Number of edges in G:", G.number_of_edges())