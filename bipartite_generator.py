# %%
from re import M

from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from bipartite_statistics import average_degree_partite
import networkx as nx
import numpy as np
from scipy.integrate import solve_ivp

from draw_networks import draw_biparite_network, get_forest_projected_tree_network, parse_forestdata_to_graph, visualize_carbon_network
from emn_model import diffusion_dynamics, diameter_to_cohort

F = parse_forestdata_to_graph()

trees, fungi = nx.bipartite.sets(F)
print(F)

# %%
print(f"trees: {trees}")
print(f"Fungae: {fungi}")


draw_biparite_network(F, trees)

avg_degrees = average_degree_partite(F, fungi)
print(f"avg degree(t,f): {avg_degrees}")

F.nodes()

# %%
G = nx.algorithms.bipartite.random_graph(len(trees), len(fungi), 0.4)
fake_trees, fake_fungi = nx.bipartite.sets(G)

G_tree = nx.bipartite.weighted_projected_graph(G, fake_trees)

# get the properties of the network like size.


fake_avg_degrees = average_degree_partite(G, fake_fungi)
print(f"avg degree(t,f): {fake_avg_degrees}")
avg_degrees = average_degree_partite(F, fungi)
print(f"avg degree(t,f): {avg_degrees}")
# draw_biparite_network(G, fake_trees)

# %%
# we want to generate multiple networks and test them for our dynamics
# What are the requirenents for this?
# 1. Generates a scale free network that conforms to the properties of the dynamics
# 2. Generic function that:
#   1. takes a network
#   2. computes general statistics
#   3. computes the dynamics
#   4. takes a set of aggregation funtions that analyse the results of the diffusion dynamics.



# based on the degree you base the amount of carbon and roots are 10% of that with noise
#
# %%
"""
The generation of a scale free network that os compatible with our diffusion model needs the following properties:
Diameter: probably a log function of degree
amount of carbon reserve: degree
amount of carbon in roots: degree
"""

# TODO: Scale based on max diameter from forest data
def generate_barabasi_forest(N: int, m: int) -> nx.Graph:
    G: nx.Graph = nx.barabasi_albert_graph(N, m)
    for n in G.nodes():
        degree = G.degree(n)
        diameter = np.sqrt(degree)
        G.nodes[n]["diameter"] = diameter
        G.nodes[n]["carbon_value"] = degree
        G.nodes[n]["cohort"] = diameter_to_cohort(diameter)
    return G

G = generate_barabasi_forest(100, 1)
visualize_carbon_network(G)


# TODO: generate barabasi starting with initial graph 

# TODO: generate 

