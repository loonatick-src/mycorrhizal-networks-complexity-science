# %%
from re import M

from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from bipartite_statistics import average_degree_partite
import networkx as nx
import numpy as np
from draw_networks import draw_biparite_network, get_forest_projected_tree_network, parse_forestdata_to_graph, visualize_carbon_network

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

def generate_scalefree_forest(N: int, m: int) -> nx.Graph:
    G: nx.Graph = nx.barabasi_albert_graph(N, m)
    for n in G.nodes():
        degree = G.degree(n)
        G.nodes[n]["diameter"] = np.sqrt(degree)
        G.nodes[n]["carbon_value"] = degree

    return G

G = generate_scalefree_forest(100, 1)
visualize_carbon_network(G)



# %%

def compute_graph_statistics(G: nx.Graph)-> dict:
    stat_funcs = [   
        nx.degree_centrality,
        nx.betweenness_centrality,
        nx.average_node_connectivity,
        nx.diameter,
    ]
    
    stats = {}
    for func in stat_funcs:
        stats[func.__name__] = func(G)
    
    return stats

def generic_analysis_func(G):
    print("not implemented")
    # The idea here is to measure for example the number of healthy growing trees or something we expect to change or show dynamics
    # the general network statistics wont change as we currently ddont change the network structure based on the network
    return 0 # TODO

def run_diffusion_model(G):
    # Do some diffusive porn stuf with the Graph
    x = 0 # TODO @Chaitanya does his thing

def run_experiment(G, analysis_funcs=[generic_analysis_func]):
    # collect initial statistics
    stats = compute_graph_statistics(G)
    
    # run the confusion model 
    data = run_diffusion_model(G)
    
    return { "stats": stats,
            "analysis": analyse(G, data, analysis_funcs),
            "sim_data": data}

def analyse(G, analysis_funcs, sim_data):
    analysis_stats = {}
    # perform analysis functions
    for func in analysis_funcs:
        analysis_stats[func.__name__] = func(G, sim_data)
        
    return analysis_stats


def run_experiments(n_min, n_max, analysis_funcs=[generic_analysis_func]):
    # this is used to initialize the barabasi network for ba_forest
    forest_graph = get_forest_projected_tree_network()
    
    n_nodes = np.linspace(n_min, n_max)
    data = {}
    
    data["ba"] = {"experiments":[],"n":[]}
    data["ba_forest"] = {"experiments":[],"n":[]} 
    data["ra"] = {"experiments":[],"n":[]}
    
    for n in n_nodes:
        G = nx.barabasi_albert_graph(n, 2)
        result = run_experiment(G)
        data["ba"]["experiments"].append(result)
        data["ba"]["n"].append(n)
        
        G = nx.barabasi_albert_graph(n, 2, initial_graph=forest_graph)
        result = run_experiment(G)
        data["ba_forest"]["experiments"].append(result)
        data["ba_forest"]["n"].append(n)
        
        G = nx.random_regular_graph(n, 2)
        result = run_experiment(G)
        data["ra"]["experiments"].append(result)
        data["ra"]["n"].append(n)
        
    return data

def run_modifying_graph_experiment(G:nx.Graph, modifying_func, mod_args, analysis_funcs=[generic_analysis_func]):
    
    modifying_func(G, mod_args)
    after_stats = compute_graph_statistics(G)
    sim_data = run_diffusion_model(G)
    
    analyse_stats = analyse(G, analysis_funcs)
    
    return {
        "stats": after_stats,
        "analysis": analyse_stats,
        "sim_data": sim_data
    }


    
    
    
    
    
    