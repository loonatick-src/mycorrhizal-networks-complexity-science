# %%
import networkx as nx
import numpy as np

from emn_model import diameter_to_cohort

# %%
"""
The generation of a scale free network that os compatible with our diffusion model needs the following properties:
Diameter: probably a log function of degree
amount of carbon reserve: degree
amount of carbon in roots: degree
"""

# TODO: Scale based on max diameter from forest data
def generate_barabasi_forest(n_nodes: int, m: int, ) -> nx.Graph:
    G: nx.Graph = nx.barabasi_albert_graph(n_nodes, m)
    
    for n in G.nodes():
        degree = G.degree(n)
        diameter = np.sqrt(degree) # TODO: Discuss the diameter scalar with the team
        G.nodes[n]["diameter"] = diameter
        G.nodes[n]["carbon_value"] = degree
        G.nodes[n]["cohort"] = diameter_to_cohort(diameter)
        
    return G

def generate_barabasi_forest_from_forest(n_nodes: int, m: int, forest_graph) -> nx.Graph:
    G: nx.Graph = nx.barabasi_albert_graph(n_nodes, m, initial_graph=forest_graph)
    for n in G.nodes():
        # node_dict = G.nodes[n]
        if G.nodes[n].get('diameter') is None:
            degree = G.degree(n)
            diameter = np.sqrt(degree) # TODO: Discuss the diameter scalar with the team
            G.nodes[n]["diameter"] = diameter
            G.nodes[n]["carbon_value"] = degree
            G.nodes[n]["cohort"] = diameter_to_cohort(diameter)
        
    return G

# TODO: Discuss: Random graph has uniform degree, thi will not create cohorts. Maybe other type of network?
def generate_random_regular_graph(n_nodes, degree):
    G = nx.random_regular_graph(d=degree, n=n_nodes)
    for n in G.nodes():
        degree = G.degree(n)
        diameter = np.sqrt(degree) # TODO: Discuss the diameter scalar with the team
        G.nodes[n]["diameter"] = diameter
        G.nodes[n]["carbon_value"] = degree
        G.nodes[n]["cohort"] = diameter_to_cohort(diameter)
        
    return G

