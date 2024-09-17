import networkx as nx

if __name__ == '__main__':
    G = nx.Graph()
    G.add_nodes_from([1,2,3,4,5])
    G.add_edges_from([(1,2),(2,4)])
    print(G.edges)
    Q :nx.Graph= G.subgraph([1,2])
    Q.remove_edge(1,2)
    print(G.edges)