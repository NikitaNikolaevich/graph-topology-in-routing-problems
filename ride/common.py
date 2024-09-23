import pickle
import networkx as nx
from ride.file_name_generator import generate_new_name
from ride.utils import extract_cluster_list_subgraph
import folium

import time
import numpy as np
from tqdm import tqdm


def find_path(
        layer,
        from_node: int,
        to_node: int,
        alg='dijkstra') -> tuple[float, list[int]]:
    """
    Find the shortest path between two nodes in a graph using a specified algorithm.

    Parameters
    ----------
    layer : GraphLayer
        The graph layer to search in.
    from_node : int
        The node to start the search from.
    to_node : int
        The node to search for.
    alg : str, optional
        The algorithm to use for the search (default is 'dijkstra'). Can be 'dijkstra', 'bidirectional', or 'astar'.

    Returns
    -------
    tuple[float, list[int]]
        A tuple containing the length of the shortest path and the path itself.

    Raises
    ------
    nx.NetworkXNoPath
        If no path is found between the nodes.
    """
    from_d = layer.graph.nodes[from_node]
    to_d = layer.graph.nodes[to_node]

    from_cluster = from_d['cluster']
    to_cluster = to_d['cluster']

    def h(a, b):
        # print(a, b)
        da = layer.graph.nodes[a]
        db = layer.graph.nodes[b]
        return ((da['x'] - db['x']) ** 2 + (da['y'] - db['y']) ** 2) ** 0.5 / 360 * 2 * np.pi * 6371.01 * 1000

    if from_cluster == to_cluster:
        try:
            g = extract_cluster_list_subgraph(layer.graph, [to_cluster], layer.communities)
            if alg == 'dijkstra':
                return nx.single_source_dijkstra(g, from_node, to_node, weight='length')
            if alg == 'bidirectional':
                return nx.bidirectional_dijkstra(g, from_node, to_node, weight='length')
            if alg == 'astar':
                return [nx.astar_path_length(g, from_node, to_node, weight='length', heuristic=h)]
        except nx.NetworkXNoPath as e:
            print('No path found in one cluster')
            raise e

    from_center = layer.cluster_to_center[from_cluster]
    to_center = layer.cluster_to_center[to_cluster]

    try:
        start = time.time()
        g = layer.centroids_graph
        path = []
        if alg == 'dijkstra':
            path = nx.single_source_dijkstra(g, from_center, to_center, weight='length')
        if alg == 'bidirectional':
            path = nx.bidirectional_dijkstra(g, from_center, to_center, weight='length')
        if alg == 'astar':
            path = [nx.astar_path_length(g, from_center, to_center, weight='length', heuristic=h)]
        end = time.time()
        step1 = end - start
    except nx.NetworkXNoPath as e:
        print('No path found in clusters')
        raise e

    start = time.time()
    cls = set()
    cls.add(to_cluster)
    for u in path[1]:
        c = layer.graph.nodes[u]['cluster']
        cls.add(c)

    g = extract_cluster_list_subgraph(layer.graph, cls, layer.communities)
    end = time.time()
    step2 = end - start
    try:
        start = time.time()

        if alg == 'dijkstra':
            path = nx.single_source_dijkstra(g, from_node, to_node, weight='length')
        if alg == 'bidirectional':
            path = nx.bidirectional_dijkstra(g, from_node, to_node, weight='length')
        if alg == 'astar':
            path = [nx.astar_path_length(g, from_node, to_node, weight='length', heuristic=h)]
        end = time.time()
        step3 = end - start
        # tqdm.write(f"""
        # step1: {step1}
        # step2: {step2}
        # step3: {step3}
        # """)
        return path
    except nx.NetworkXNoPath as e:
        print(nx.is_connected(g))
        print('No path in cluster subgraph')
        raise e


def find_path_length(
        layer,
        from_node: int,
        to_node: int,
        alg='dijkstra') -> float:
    """
    Find the length of the shortest path between two nodes in a graph using a specified algorithm.

    Parameters
    ----------
    layer : GraphLayer
        The graph layer to search in.
    from_node : int
        The node to start the search from.
    to_node : int
        The node to search for.
    alg : str, optional
        The algorithm to use for the search (default is 'dijkstra'). Can be 'dijkstra', 'bidirectional', or 'astar'.

    Returns
    -------
    tuple[float, float, float, float]
        A tuple containing the length of the shortest path, and the times taken for each step of the algorithm.

    Raises
    ------
    nx.NetworkXNoPath
        If no path is found between the nodes.
    """
    from_d = layer.graph.nodes[from_node]
    to_d = layer.graph.nodes[to_node]

    from_cluster = from_d['cluster']
    to_cluster = to_d['cluster']

    def h(a, b):
        # print(a, b)
        da = layer.graph.nodes[a]
        db = layer.graph.nodes[b]
        return ((da['x'] - db['x']) ** 2 + (da['y'] - db['y']) ** 2) ** 0.5 / 360 * 2 * np.pi * 6371.01 * 1000

    if from_cluster == to_cluster:
        try:
            g = extract_cluster_list_subgraph(layer.graph, [to_cluster], layer.communities)
            if alg == 'dijkstra':
                return nx.single_source_dijkstra(g, from_node, to_node, weight='length')[0], 0,0,0
            if alg == 'bidirectional':
                return nx.bidirectional_dijkstra(g, from_node, to_node, weight='length')[0], 0,0,0
            if alg == 'astar':
                return nx.astar_path_length(g, from_node, to_node, weight='length', heuristic=h), 0,0,0
        except nx.NetworkXNoPath as e:
            print('No path found in one cluster')
            return -1

    from_center = layer.cluster_to_center[from_cluster]
    to_center = layer.cluster_to_center[to_cluster]

    try:
        start = time.time()

        g = layer.centroids_graph
        path = []
        if alg == 'dijkstra':
            path = nx.single_source_dijkstra(g, from_center, to_center, weight='length')[1]
        if alg == 'bidirectional':
            path = nx.bidirectional_dijkstra(g, from_center, to_center, weight='length')[1]
        if alg == 'astar':
            path = nx.astar_path(g, from_center, to_center, weight='length', heuristic=h)
        end = time.time()
        step1 = end - start
    except nx.NetworkXNoPath as e:
        print('No path found in clusters')
        return -1
    start = time.time()
    cls2next = {}
    cls = set()
    cls.add(to_cluster)
    prev = from_cluster
    for u in path:
        du = layer.graph.nodes[u]
        c = du['cluster']
        if c != prev:
            cls2next[prev] = u
        prev = c
        cls.add(c)
    cls2next[prev] = to_node

    def h1(a, b):
        da = layer.graph.nodes[a]
        db = layer.graph.nodes[cls2next[da['cluster']]]
        # db = layer.graph.nodes[b]
        return ((da['x'] - db['x']) ** 2 + (da['y'] - db['y']) ** 2) ** 0.5 / 360 * 2 * np.pi * 6371.01 * 1000

    g = extract_cluster_list_subgraph(layer.graph, cls, layer.communities)
    end = time.time()
    step2 = end - start
    try:
        start = time.time()

        if alg == 'dijkstra':
            path = nx.single_source_dijkstra(g, from_node, to_node, weight='length')[0]
        if alg == 'bidirectional':
            path = nx.bidirectional_dijkstra(g, from_node, to_node, weight='length')[0]
        if alg == 'astar':
            path = nx.astar_path_length(g, from_node, to_node, weight='length', heuristic=h1)
        end = time.time()
        step3 = end - start
        # tqdm.write(f"""
        # step1: {step1}
        # step2: {step2}
        # step3: {step3}
        # """)
        return path, step1,step2,step3
    except nx.NetworkXNoPath as e:
        print(nx.is_connected(g))
        print('No path in cluster subgraph')
        return -1



class CentroidResult:
    """
        Represents the result of a centroid calculation.

        Attributes
        ----------
        resolution : float
            The resolution of the centroid calculation
        centroid_nodes : int
            The number of nodes in the centroid
        centroid_edges : int
            The number of edges in the centroid
        centroid_density : float
            The density of the centroid (calculated as 2 * centroid_edges / (centroid_nodes * (centroid_nodes - 1)))
        alpha : float
            Еhe ratio of the number of clusters to the number of nodes
        speed_up : list[float]
            The acceleration relative to the basic solution
        errors : list[float]
            The error values relative to the basic solution
        absolute_time : list[float]
            The absolute time values for the centroid calculation
        absolute_err : list[float]
            The absolute error values for the centroid calculation

        Parameters
        ----------
        resolution : float
            The resolution of the centroid calculation
        centroid_nodes : int
            The number of nodes in the centroid
        centroid_edges : int
            The number of edges in the centroid
        alpha : float
            Еhe ratio of the number of clusters to the number of nodes
    """

    def __init__(self, resolution: float,
                 centroid_nodes: int,
                 centroid_edges: int,
                 alpha: float
                 ):
        self.resolution: float = resolution
        self.centroid_nodes: int = centroid_nodes
        self.centroid_edges: int = centroid_edges
        if centroid_nodes == 1:
            self.centroid_density = 0
        else:
            self.centroid_density = 2 * centroid_edges / (centroid_nodes * (centroid_nodes - 1))
        self.alpha: float = alpha
        self.speed_up: list[float] = []
        self.errors: list[float] = []
        self.absolute_time: list[float] = []
        self.absolute_err: list[float] = []


class CityResult:
    """
        Represents the result of a city calculation.

        Attributes
        ----------
        name : str
            The name of the city
        name_suffix : str
            The suffix of the city name
        id : str
            The ID of the city
        nodes : int
            The number of nodes in the city graph
        edges : int
            The number of edges in the city graph
        density : float
            The density of the city graph (calculated as 2 * edges / (nodes * (nodes - 1)))
        points_results : list[CentroidResult]
            The list of centroid results for the city

        Parameters
        ----------
        name : str
            The name of the city
        name_suffix : str
            The suffix of the city name
        city_id : str
            The ID of the city
        nodes : int
            The number of nodes in the city graph
        edges : int
            The number of edges in the city graph

        Methods
        -------
        save()
            Saves the city result to a file
    """

    def __init__(self,
                 name: str,
                 name_suffix: str,
                 city_id: str,
                 nodes: int,
                 edges: int
                 ):
        self.name = name
        self.name_suffix: str = name_suffix
        self.id: str = city_id
        self.nodes: int = nodes
        self.edges: int = edges
        self.density: float = 2 * edges / (nodes * (nodes - 1))
        self.points_results: list[CentroidResult] = []

    def save(self):
        file_name = generate_new_name(self.name + self.name_suffix + '.pkl')
        with open(file_name, 'wb') as fp:
            pickle.dump(self, fp)
            fp.close()


class GraphLayer:
    """
        Represents a graph layer with communities, cluster information, and centroids.

        Attributes
        ----------
        cluster_to_center : dict[int, int]
            Mapping of cluster IDs to their corresponding center nodes
        cluster_to_bridge_points : dict[int, set[int]]
            Mapping of cluster IDs to their corresponding bridge points
        cluster_to_neighboring_cluster : dict[int, set[int]]
            Mapping of cluster IDs to their neighboring clusters
        communities : list[set[int]]
            List of communities in the graph
        centroids_graph : nx.Graph
            Graph of centroids
        graph : nx.Graph
            Original graph

        Parameters
        ----------
        graph : nx.Graph
            Original graph
        communities : list[set[int]]
            List of communities in the graph
        cluster_to_neighboring_cluster : dict[int, set[int]]
            Mapping of cluster IDs to their neighboring clusters
        cluster_to_bridge_points : dict[int, set[int]]
            Mapping of cluster IDs to their bridge points
        cluster_to_center : dict[int, int]
            Mapping of cluster IDs to their corresponding center nodes
        centroids_graph : nx.Graph
            Graph of centroids
    """

    def __init__(self,
                 graph: nx.Graph,
                 communities: list[set[int]],
                 cluster_to_neighboring_cluster: dict[int, set[int]],
                 cluster_to_bridge_points: dict[int, set[int]],
                 cluster_to_center: dict[int, int],
                 centroids_graph: nx.Graph
                 ):
        self.cluster_to_center = cluster_to_center
        self.cluster_to_bridge_points = cluster_to_bridge_points
        self.cluster_to_neighboring_cluster = cluster_to_neighboring_cluster
        self.communities = communities
        self.centroids_graph = centroids_graph
        self.graph = graph

    def draw_graph(self, visible=False):
        m = folium.Map(zoom_start=10)

        for node in self.graph.nodes():
            folium.Circle([self.graph.nodes[node]['y'], self.graph.nodes[node]['x']], popup=str(node), fill=True).add_to(m)
        for edge in self.graph.edges():
            folium.PolyLine([[self.graph.nodes[edge[0]]['y'], self.graph.nodes[edge[0]]['x']], [self.graph.nodes[edge[1]]['y'], self.graph.nodes[edge[1]]['x']]], color='blue', weight=2).add_to(m)
        
        m.save('graph.html')

        if visible:
           return m


    def draw_path(self, nodes: list[int], x: list[float], y: list[float], visible=False):
        m = folium.Map(location=[y[0], x[0]], zoom_start=10)

        for i in range(len(nodes) - 1):
            folium.PolyLine([[y[i], x[i]], [y[i + 1], x[i + 1]]], color='red', weight=5).add_to(m)
        # for node in nodes:
        #     folium.Marker([y[node], x[node]], popup=str(node)).add_to(m)

        m.save('path.html')
        
        if visible:
           return m

    def find_path(self, from_node: int, to_node: int, alg: str = 'dijkstra', draw_path=False, visible=False) -> tuple[float, list[int]]:
        distance, nodes = find_path(self, from_node, to_node, alg)
        x = [self.graph.nodes[node]['x'] for node in nodes]
        y = [self.graph.nodes[node]['y'] for node in nodes]
        if draw_path:
            maps = self.draw_path(nodes, x, y, visible=visible)
            return distance, nodes, maps
        return distance, nodes