import os
import random
import itertools

import networkx as nx
import osmnx as ox
from tqdm import tqdm
import pandas as pd
import math

class DataGetter:

    @staticmethod
    def _geocode_to_gdf(id, by_osmid=True):
        gdf = ox.geocode_to_gdf(id, by_osmid=by_osmid)
        return gdf

    @staticmethod
    def _get_polygon_boundary(gdf):
        polygon_boundary = gdf.unary_union
        return polygon_boundary

    @staticmethod
    def _get_graph_from_polygon(polygon_boundary, network_type='drive', simplify=True):
        graph = ox.graph_from_polygon(polygon_boundary, network_type=network_type, simplify=simplify)
        return graph

    @staticmethod
    def _create_graph(graph):
        G = nx.Graph(graph)
        return G

    @staticmethod
    def _create_empty_graph():
        H = nx.Graph()
        return H

    @staticmethod
    def _add_nodes_to_graph(G, H):
        for u, d in G.nodes(data=True):
            H.add_node(u, x=d['x'], y=d['y'])

    @staticmethod
    def _add_edges_to_graph(G, H):
        for u, v, d in G.edges(data=True):
            H.add_edge(u, v, length=d['length'])

    @staticmethod
    def _create_directory(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def _write_graph_to_file(H, file_path):
        nx.write_graphml(H, file_path)

    @staticmethod
    def _print_save_message(id, file_path):
        print(f"Граф с id {id} был сохранён в {file_path}")

    @staticmethod
    def download_graph(id, simplify=True):
        gdf = DataGetter._geocode_to_gdf(id)
        polygon_boundary = DataGetter._get_polygon_boundary(gdf)
        graph = DataGetter._get_graph_from_polygon(polygon_boundary, simplify=simplify)
        G = DataGetter._create_graph(graph)
        H = DataGetter._create_empty_graph()
        DataGetter._add_nodes_to_graph(G, H)
        DataGetter._add_edges_to_graph(G, H)
        directory = "graphs"
        DataGetter._create_directory(directory)
        file_path = os.path.join(directory, f"graph_{id}.graphml")
        DataGetter._write_graph_to_file(H, file_path)
        DataGetter._print_save_message(id, file_path)
        H.graph['id'] = id
        return H
    
class DataGenerator:

    @staticmethod
    def _percent(l, p=0.1, max_points=10):
        res = int(len(l) * p) if len(l) * p >= 1 else 1
        if res > max_points:
            return max_points
        else:
            return res

    @staticmethod
    def _get_path_length_dict(G_cl):
        d = dict(nx.all_pairs_dijkstra_path_length(G_cl))
        return d

    @staticmethod
    def _create_dataframe(d):
        df = pd.DataFrame(d)
        df = df.reindex(sorted(df.columns), axis=1)
        df = df.sort_index()
        return df

    @staticmethod
    def _get_graph_path_length(df):
        graph_path_length = df.values
        return graph_path_length

    @staticmethod
    def _get_nodes_pairs(graph_path_length, min_length=5):
        res = []
        for com_1 in range(len(graph_path_length)):
            for com_2 in range(com_1 + 1, len(graph_path_length)):
                if graph_path_length[com_1][com_2] >= min_length:
                    res.append((com_1, com_2))
        return res

    @staticmethod
    def _get_nodes_from_clusters(clusters, com, p=0.1, max_points=2):
        return random.sample(clusters[com],
                            k=DataGenerator._percent(clusters[com], p, max_points))

    @staticmethod
    def _get_all_nodes_pairs(list_1, list_2):
        return list(itertools.product(list_1, list_2))

    @staticmethod
    def sample_nodes_for_experiment(G_cl, clusters, min_length=5, p=0.1, max_points=2):
        d = DataGenerator._get_path_length_dict(G_cl)
        df = DataGenerator._create_dataframe(d)
        graph_path_length = DataGenerator._get_graph_path_length(df)
        nodes_pairs = DataGenerator._get_nodes_pairs(graph_path_length, min_length)
        res = []

        for com_1, com_2 in nodes_pairs:
            list_1 = DataGenerator._get_nodes_from_clusters(clusters, com_1, p, max_points)
            list_2 = DataGenerator._get_nodes_from_clusters(clusters, com_2, p, max_points)
            all_lists = DataGenerator._get_all_nodes_pairs(list_1, list_2)

            res.extend(list(all_lists))
        return res
    
    @staticmethod
    def formula_centroid(v0: int, e0: int, e1: int, k: float) -> float:
        """Calculate the formula for the centroid"""
        return e1 * math.log(k*v0) + v0 + (e0/math.sqrt(k*v0) * (math.log(math.sqrt(v0/k))))

    @staticmethod
    def formula_louven(H: nx.Graph) -> float:
        """Calculate the formula for Louvain algorithm"""
        m = len(H.nodes())
        n = len(H.edges())
        return (n*math.log(n) + m*math.log(n))