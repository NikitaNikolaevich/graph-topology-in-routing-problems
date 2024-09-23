import os
import random
import itertools

import networkx as nx
import osmnx as ox
from tqdm import tqdm
import pandas as pd
import math
from typing import Union

class DataGetter:
    """
    A class for getting and manipulating graph data.
    """

    @staticmethod
    def _geocode_to_gdf(id, by_osmid=True):
        """
        Geocode an ID to a GeoDataFrame.

        Parameters
        ----------
        id : str or int
            The ID to geocode.
        by_osmid : bool, optional
            Whether to geocode by OSM ID (default is True).

        Returns
        -------
        gdf : GeoDataFrame
            The GeoDataFrame corresponding to the ID.
        """
        gdf = ox.geocode_to_gdf(id, by_osmid=by_osmid, )
        return gdf

    @staticmethod
    def _get_polygon_boundary(gdf):
        """
        Get the polygon boundary of a GeoDataFrame.

        Parameters
        ----------
        gdf : GeoDataFrame
            The GeoDataFrame to get the boundary of.

        Returns
        -------
        polygon_boundary : Polygon
            The polygon boundary of the GeoDataFrame.
        """
        polygon_boundary = gdf.unary_union
        return polygon_boundary

    @staticmethod
    def _get_graph_from_polygon(polygon_boundary, network_type='drive', simplify=True):
        """
        Get a graph from a polygon boundary.

        Parameters
        ----------
        polygon_boundary : Polygon
            The polygon boundary to get the graph from.
        network_type : str, optional
            The type of network to get (default is 'drive').
        simplify : bool, optional
            Whether to simplify the graph (default is True).

        Returns
        -------
        graph : Graph
            The graph corresponding to the polygon boundary.
        """
        graph = ox.graph_from_polygon(polygon_boundary, network_type=network_type, simplify=simplify)
        return graph

    @staticmethod
    def _create_graph(graph):
        """
        Create a new graph from an existing graph.

        Parameters
        ----------
        graph : Graph
            The graph to create a new graph from.

        Returns
        -------
        G : Graph
            The new graph.
        """
        # G = nx.MultiDiGraph(graph)
        G = nx.Graph(graph)
        return G

    @staticmethod
    def _create_empty_graph(id):
        """
        Create an empty graph with a given ID.

        Parameters
        ----------
        id : str or int
            The ID of the graph.

        Returns
        -------
        H : Graph
            The empty graph.
        """
        # H = nx.MultiDiGraph()
        H = nx.Graph()
        H.graph['id'] = id
        return H

    @staticmethod
    def _add_nodes_to_graph(G, H):
        """
        Add nodes from one graph to another.

        Parameters
        ----------
        G : Graph
            The graph to add nodes from.
        H : Graph
            The graph to add nodes to.
        """

        for u, d in G.nodes(data=True):
            H.add_node(u, x=d['x'], y=d['y'])

    @staticmethod
    def _add_edges_to_graph(G, H, weight='length'):
        """
        Add edges from one graph to another.

        Parameters
        ----------
        G : Graph
            The graph to add edges from.
        H : Graph
            The graph to add edges to.
        weight : str, optional
            The weight of the edges (default is 'length').
        """
        for u, v, d in G.edges(data=True):
            H.add_edge(u, v, length=d[weight])

    @staticmethod
    def _create_directory(directory):
        """
        Create a directory if it does not exist.

        Parameters
        ----------
        directory : str
            The directory to create.
        """

        if not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def _write_graph_to_file(H, file_path):
        """
        Write a graph to a file.

        Parameters
        ----------
        H : Graph
            The graph to write.
        file_path : str
            The path to the file.
        """

        nx.write_graphml(H, file_path)

    @staticmethod
    def _print_save_message(id, file_path):
        """
        Print a message indicating that a graph has been saved.

        Parameters
        ----------
        id : str or int
            The ID of the graph.
        file_path : str
            The path to the file.
        """
        print(f"Граф с id {id} был сохранён в {file_path}")

    @staticmethod
    def _justify_input_id(id):
        """
        Justify an input ID by adding 'R' to the beginning if necessary.

        Parameters
        ----------
        id : str or int
            The ID to justify.

        Returns
        -------
        id : str
            The justified ID.
        """

        #R -- means relation, which is a polygon of the selected city
        if isinstance(id,str):
            if "R" in id:
                pass
            else:
                id = "R" + id
        else:
            id = "R" + str(id)
        return id

    @staticmethod
    def download_graph(id: Union[int, str], simplify=True, save_on_device=False):
        """
        Download a graph from OpenStreetMap.

        Parameters
        ----------
        id : str or int
            The ID of the graph to download.
        simplify : bool, optional
            Whether to simplify the graph (default is True).
        save_on_device : bool, optional
            Whether to save the graph on the device (default is False).

        Returns
        -------
        H : Graph
            The downloaded graph.
        """
        id = DataGetter._justify_input_id(id)
        gdf = DataGetter._geocode_to_gdf(id)
        polygon_boundary = DataGetter._get_polygon_boundary(gdf)
        graph = DataGetter._get_graph_from_polygon(polygon_boundary, simplify=simplify)
        G = DataGetter._create_graph(graph)
        H = DataGetter._create_empty_graph(id)
        DataGetter._add_nodes_to_graph(G, H)
        DataGetter._add_edges_to_graph(G, H)
        directory = "graphs"
        DataGetter._create_directory(directory)
        file_path = os.path.join(directory, f"graph_{id}.graphml")

        if save_on_device:
            DataGetter._write_graph_to_file(H, file_path)
            DataGetter._print_save_message(id, file_path)

        return H
    
    @staticmethod
    def load_graph(path):
        """Load a graph from a file.

        Parameters
        ----------
        path : str
            The path to the file.

        Returns
        -------
        H : Graph
            The loaded graph.

        Raises
        ------
        ValueError
            If the file is not a.graphml file.
        """

        if path.endswith('.graphml'):
            G = nx.read_graphml(path)
            H = DataGetter._create_empty_graph(id)
            DataGetter._add_nodes_to_graph(G, H)
            DataGetter._add_edges_to_graph(G, H)
            return H
        else:
            raise ValueError("Invalid file type. Please provide a .graphml file.")
        
class DataGenerator:
    """
    A class for generating data for experiments.
    """

    @staticmethod
    def _percent(l, p=0.1, max_points=10):
        """
        Calculate the percentage of elements in a list.

        Parameters
        ----------
        l : list
            The list to calculate the percentage for.
        p : float, optional
            The percentage (default is 0.1).
        max_points : int, optional
            The maximum number of points (default is 10).

        Returns
        -------
        int
            The calculated percentage.
        """
        res = int(len(l) * p) if len(l) * p >= 1 else 1
        if res > max_points:
            return max_points
        else:
            return res

    @staticmethod
    def _get_path_length_dict(G_cl):
        """
        Get the path length dictionary for a graph.

        Parameters
        ----------
        G_cl : Graph
            The graph to get the path length dictionary for.

        Returns
        -------
        dict
            The path length dictionary.
        """
        d = dict(nx.all_pairs_dijkstra_path_length(G_cl))
        return d

    @staticmethod
    def _create_dataframe(d):
        """
        Create a DataFrame from a dictionary.

        Parameters
        ----------
        d : dict
            The dictionary to create the DataFrame from.

        Returns
        -------
        DataFrame
            The created DataFrame.
        """

        df = pd.DataFrame(d)
        df = df.reindex(sorted(df.columns), axis=1)
        df = df.sort_index()
        return df

    @staticmethod
    def _get_graph_path_length(df):
        """
        Get the graph path length from a DataFrame.

        Parameters
        ----------
        df : DataFrame
            The DataFrame to get the graph path length from.

        Returns
        -------
        array
            The graph path length.
        """

        graph_path_length = df.values
        return graph_path_length

    @staticmethod
    def _get_nodes_pairs(graph_path_length, min_length=5):
        """
        Get the nodes pairs from a graph path length.

        Parameters
        ----------
        graph_path_length : array
            The graph path length.
        min_length : int, optional
            The minimum length (default is 5).

        Returns
        -------
        list
            The nodes pairs.
        """

        res = []
        for com_1 in range(len(graph_path_length)):
            for com_2 in range(com_1 + 1, len(graph_path_length)):
                if graph_path_length[com_1][com_2] >= min_length:
                    res.append((com_1, com_2))
        return res

    @staticmethod
    def _get_nodes_from_clusters(clusters, com, p=0.1, max_points=2):
        """
        Get the nodes from a cluster.

        Parameters
        ----------
        clusters : dict
            The clusters to get the nodes from.
        com : int
            The cluster to get the nodes from.
        p : float, optional
            The percentage (default is 0.1).
        max_points : int, optional
            The maximum number of points (default is 2).

        Returns
        -------
        list
            The nodes.
        """

        return random.sample(clusters[com],
                            k=DataGenerator._percent(clusters[com], p, max_points))

    @staticmethod
    def _get_all_nodes_pairs(list_1, list_2):
        """
        Get all nodes pairs from two lists.

        Parameters
        ----------
        list_1 : list
            The first list.
        list_2 : list
            The second list.

        Returns
        -------
        list
            The nodes pairs.
        """

        return list(itertools.product(list_1, list_2))

    @staticmethod
    def sample_nodes_for_experiment(G_cl, clusters, min_length=5, p=0.1, max_points=2):
        """
        Sample nodes for an experiment.

        Parameters
        ----------
        G_cl : Graph
            The graph to sample nodes from.
        clusters : dict
            The clusters to sample nodes from.
        min_length : int, optional
            The minimum length (default is 5).
        p : float, optional
            The percentage (default is 0.1).
        max_points : int, optional
            The maximum number of points (default is 2).

        Returns
        -------
        list
            The sampled nodes.
        """

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
        """
        Calculate the formula for the centroid.

        Parameters
        ----------
        v0 : int
            The number of vertices.
        e0 : int
            The number of edges.
        e1 : int
            The number of edges in the cluster.
        k : float
            The ratio of clusters to vertices.

        Returns
        -------
        float
            The calculated formula.
        """
        return e1 * math.log(k*v0) + v0 + (e0/math.sqrt(k*v0) * (math.log(math.sqrt(v0/k))))

    @staticmethod
    def formula_louven(H: nx.Graph) -> float:
        """
        Calculate the formula for the Louvain algorithm.

        Parameters
        ----------
        H : nx.Graph
            The graph to calculate the formula for.

        Returns
        -------
        float
            The calculated formula.
        """

        m = len(H.nodes())
        n = len(H.edges())
        return (n*math.log(n) + m*math.log(n))