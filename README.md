<img src=https://github.com/NikitaNikolaevich/graph-topology-in-routing-problems/blob/main/images/speed.png/>

# CORP (Clustering Optimiser for Routing problem)  or RIDE (Rapid infra-cluster dijkstra enhancer) library

The CORP\RIDE or Clustering Optimiser for Routing problem is a python library for accelerating Deikstra task on any graphs with hierarchical method involving solving a problem on simpler graphs with further combining solutions into a common one. More information about method ine can find soon in corresponding _article_.

<img src=https://github.com/NikitaNikolaevich/graph-topology-in-routing-problems/blob/main/images/milan.png width="600"/>

# Installing

(RIDE library) -- Rapid infra-cluster dijkstra enhancer

to install via pip without listing on pipy do: 
```pip install git+https://github.com/<username>/<repository>.git@main#egg=gsber```

# Quick start

First u need for a quick start it to choose graph u need, u can create or download it yourself or use G = DataGetter.download_graph(id="R44915") function to download graph from the Open Street Maps. Search for the object and then copy id for the DataGetter.download_graph like following picture shows:

<img src=https://github.com/NikitaNikolaevich/graph-topology-in-routing-problems/blob/main/images/osm.png width="600"/>

then all you need is a following code:


```
#downloading graph
G = DataGetter.download_graph(id="R44915")

#preprocessing
resolutions, ks, v1, e1 = GraphProcessor.search_resolutions(G, k_max=0.6)

#path finding function
output, mistakes = GraphRunner.compute_shortest_path_length_centroid(G,resolutions,testing_points,all_length,'length',output )
```

It is worth noting that this method works for both transport and abstract graphs.
