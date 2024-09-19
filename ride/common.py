import pickle
import networkx as nx
from ride.file_name_generator import generate_new_name


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


class Layer:
    """
        Represents a layer with communities and centroids.

        Attributes
        ----------
        cluster_to_center : dict[int, int]
            Mapping of cluster IDs to their corresponding center nodes
        communities : list[set[int]]
            List of communities in the layer
        centroids_graph : nx.Graph
            Graph of centroids
        graph : nx.Graph
            Original graph

        Parameters
        ----------
        graph : nx.Graph
            Original graph
        communities : list[set[int]]
            List of communities in the layer
        cluster_to_center : dict[int, int]
            Mapping of cluster IDs to their corresponding center nodes
        centroids_graph : nx.Graph
            Graph of centroids
    """

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
    """
        Represents a hierarchical structure composed of graph layers.

        Attributes
        ----------
        layers : list[GraphLayer]
            List of graph layers

        Parameters
        ----------
        layers : list[GraphLayer]
            List of graph layers
    """
    def __init__(self, layers: list[GraphLayer]):
        self.layers = layers
