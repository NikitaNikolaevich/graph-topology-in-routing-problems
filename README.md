<!--
<img src=https://github.com/NikitaNikolaevich/graph-topology-in-routing-problems/blob/main/images/speed.png/>
-->

# RIDE
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![PythonVersion](https://img.shields.io/badge/python-3.11-blue)](https://pypi.org/project/scikit-learn/)
[![Documentation Status](https://readthedocs.org/projects/graph-topology-in-routing-problems/badge/?version=latest)](https://graph-topology-in-routing-problems.readthedocs.io/en/latest/?badge=latest)
<!-- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/NikitaNikolaevich/graph-topology-in-routing-problems/main?urlpath=https%3A%2F%2Fgithub.com%2FNikitaNikolaevich%2Fgraph-topology-in-routing-problems%2Fblob%2Fdocs%2Fexamples%2Fexamples.ipynb) -->

<img align="right" src="docs/logo/logo.png" alt="Your Banner" width="20%">

**RIDE** (Rapid infra-cluster dijkstra enhancer) library

The RIDE is a python library for accelerating Deikstra task on any graphs with hierarchical method involving solving a problem on simpler graphs with further combining solutions into a common one. The method is based on the division of the graph into clusters. By using this division, you can eliminate many sub optimal route constructions and achieve multiple-time acceleration without significant loss of accuracy. More information about method ine can find soon in corresponding _article_.

***It is worth noting that this method works for both transport and abstract graphs.***
<!-- 
<img src=https://github.com/NikitaNikolaevich/graph-topology-in-routing-problems/blob/main/images/milan.png width="600"/>
-->

# Installing

to install via pip without listing on pipy do: 
```
!pip install git+https://github.com/NikitaNikolaevich/graph-topology-in-routing-problems.git@main
```

# Quick start

We encourage you to find suitable code snippet in the examples folder :)

# How it works:
1. Creation of a new graph based on centers of initial graph clusters

![Clustering](./images/clustering.png)

2. Computation of shortes path on a new cluster-based graph (this contraction-hierarchy based approach is obviously faster hhan straight forward calcylation of shortest path, but less accurate)

![Subgraph_path](./images/subgraph_path.png)

3. Comparison of obtained metric for error-speedup trade-off

![Subgraph_path](./images/metrics.png)

# Findings

The relationship between theoretical estimations and empirical cal-
culations. Figure 1,3 – the relationship between the maximum of acceleration γmax and the number of vertices N0 in the graph.
the relationship between the optimal value of the α∗parameter and the number of vertices N0. Figure 2 – the dependence of the maximum acceleration γmax on the graph density D (unscaled characteristic) along with the theoretical estimations, considering the equality given by D=2β0/N0.

Developed algorithm was applied for 600 cities and the following dependencies were obtained:

<div style="text-align: center;">
    <img align="left" src="./images/all_a.png" alt="Your Banner1" width="30%">
    <img align="center" src="./images/all_y.png" alt="Your Banner2" width="30%">
    <img align="right" src="./images/all_y_max.png" alt="Your Banner3" width="30%">
</div>

<!-- # Results

Explore the performance of the Hierarchical Pathfinding Algorithm compared to the classical Dijkstra algorithm through the following graphs: -->

<!-- ![Prague Graph](./images/Prague.png) -->

<!-- 
The relationship between the maximum acceleration $γ_{max}$ and the number of vertices $N_0$ in the graph.

![Max Acceleration](./images/max_acceleration.png)
-->

## Contributing

We welcome contributions! Please read our [Contributing Guidelines](CONTRIBUTING.md) for more information.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Dijkstra's Algorithm](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm)
- [Open Street Maps](https://www.openstreetmap.org)

---

For more information, check out our [documentation](https://graph-topology-in-routing-problems.readthedocs.io/en/latest/).