import networkx as nx
import numpy as np

from .emn_model import diameter_to_cohort

"""
The generation of a scale free network that is compatible
with our diffusion model APIs needs the following properties:
- Diameter: probably a log function of degree
- amount of carbon reserve: degree
- amount of carbon in roots: degree
"""


def degree_diameter_relation(min_degree, max_degree):
    min_diameter, max_diameter = 1, 55
    a = (max_diameter - min_diameter) / (max_degree - min_degree)
    b = min_diameter - a * min_degree
    return a, b


def add_randomness_to_diameter(diameter):
    def cohort_diameter_std_relation(
        x, scale): return x * scale / (x * scale + 1) * 7

    diameter = abs(
        np.random.normal(
            diameter,
            cohort_diameter_std_relation(
                diameter,
                0.2)))

    return diameter


def calc_carbon(diameter, max_diameter, stress_level=0, carbon_scalar=500):
    fraction = diameter / max_diameter
    carbon = np.tanh((fraction - 0.5) * np.pi * 2) + stress_level
    return (carbon + 1) * carbon_scalar


def set_carbon_value_for_node(n: int, G: nx.Graph, a, b):
    degree = G.degree(n)
    diameter = a * degree + b
    randomised_diameter = add_randomness_to_diameter(diameter)
    G.nodes[n]["diameter"] = randomised_diameter
    G.nodes[n]["carbon_value"] = calc_carbon(randomised_diameter, 55)
    G.nodes[n]["cohort"] = diameter_to_cohort(randomised_diameter)


def generate_barabasi_forest(n_nodes: int, m: int, seed=500) -> nx.Graph:
    # TODO This function fails magically sometimes, but not always. 
    G: nx.Graph = nx.barabasi_albert_graph(n_nodes, m, seed=seed)
    # find highest degree and the smallest degree to map the diameter to the max and the min
    degree_sequence = [d for n, d in G.degree()]
    a, b = degree_diameter_relation(degree_sequence[-1], degree_sequence[0])

    for n in G.nodes():
        set_carbon_value_for_node(n, G, a, b)

    return G


def generate_barabasi_forest_from_forest(
        n_nodes: int, m: int, forest_graph) -> nx.Graph:
    G: nx.Graph = nx.barabasi_albert_graph(
        n_nodes, m, initial_graph=forest_graph, seed=500)
    degree_sequence = [d for n, d in G.degree()]
    a, b = degree_diameter_relation(degree_sequence[-1], degree_sequence[0])

    for n in G.nodes():
        # node_dict = G.nodes[n]
        if G.nodes[n].get('diameter') is None:
            set_carbon_value_for_node(n, G, a, b)

    return G


def generate_random_graph(n_nodes, p=0.2):
    G: nx.Graph = nx.erdos_renyi_graph(n_nodes, p=p, seed=500)

    degree_sequence = [d for n, d in G.degree()]
    a, b = degree_diameter_relation(degree_sequence[-1], degree_sequence[0])

    for n in G.nodes():
        set_carbon_value_for_node(n, G, a, b)

    return G
