import pickle

import networkx as nx

from file_name_generator import generate_new_name


class CentroidResult:
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


class Layer:
    def __init__(self,
                 graph: nx.Graph,
                 communities: list[set[int]],
                 cluster_to_center: dict[int, int],
                 centroids_graph: nx.Graph
                 ):
        self.cluster_to_center = cluster_to_center
        self.communities = communities
        self.centroids_graph = centroids_graph
        self.graph = graph


class HStructure:
    def __init__(self, layers: list[GraphLayer]):
        self.layers = layers
