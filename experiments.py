import networkx as nx
import numpy as np
from scipy.integrate import solve_ivp

from forest_generator import generate_barabasi_forest, generate_barabasi_forest_from_forest, generate_random_regular_graph
from emn_model import diffusion_dynamics, generate_bipartite_network, get_clean_dataset, tree_project_network


def compute_graph_statistics(G: nx.Graph)-> dict:
    stat_funcs = [   
        nx.degree_centrality,
        nx.betweenness_centrality,
        nx.average_node_connectivity,
        nx.diameter,
        nx.eigenvector_centrality,
    ]
    
    stats = {}
    for func in stat_funcs:
        stats[func.__name__] = func(G)
    
    return stats

def sampling_groth_stats(G, data):
    roots_carbon, plant_carbon, plant_diameter =  data
    
    nodes_by_cohort = {"Sapling": [], "Maturing": [], "Established": []}
    node_list = list(G.nodes)

    for node in G.nodes:
        nodes_by_cohort[G.nodes[node]["cohort"]].append(node)
    
    for cohort, nodes in nodes_by_cohort.items():
    
        node_idxs = [node_list.index(node) for node in nodes]
        
        #Calculate Average growth
        growth = plant_diameter[node_idxs, -1] - plant_diameter[node_idxs, 0]
        avg_growth = np.mean(growth)
        
        #Calculate growth in percantages
        percentages = plant_diameter[node_idxs, -1]/plant_diameter[node_idxs, 0]*100
        percentage_mean = np.mean(percentages)
        percentage_stdev = np.std(percentages)
        
        if cohort == "Sapling": 
            n_successes = np.sum([percentages >= 200])
            result = {
                "average_growth": avg_growth,
                "percentage_mean_growth": percentage_mean,
                "percentage_stdev_growth": percentage_stdev,
                "nsuccesses": n_successes,
                "percentage_succesful_growth": n_successes/len(percentages) * 100
            }
    
    return result

def run_diffusion_model(G):
    N = len(G.nodes)

    plant_carbon_0 = np.array(list(nx.get_node_attributes(G, "carbon_value").values()))
    root_carbon_0 = 0.1 * plant_carbon_0
    plant_diameter_0 = np.array(list(nx.get_node_attributes(G, "diameter").values()))

    t_range = (0, 16_000)
    D_C = 5.0e-4 # carbon diffusion coeff
    uptake_ps = (0.55, 0.0, 55.0)
    f = 5.0e-4 # sensitive w.r.t plant carbon deposition
    k = 0.6
    g = 0.004
    c = 0.82
    rho = 0.01

    sol = solve_ivp(diffusion_dynamics, t_range, np.concatenate([root_carbon_0, plant_carbon_0, plant_diameter_0]),
                    args=(G, D_C, N, uptake_ps, f, k, c, g, rho), dense_output=True, method="BDF")

    time_steps = 300
    t = np.linspace(t_range[0], t_range[1], time_steps)
    z = sol.sol(t)

    #Rename the quantities of interest
    roots_carbon, plant_carbon, plant_diameter = z[:N, :], z[N:2*N, :], z[2*N:3*N, :] 
    
    return roots_carbon, plant_carbon, plant_diameter


def run_experiment(G, analysis_funcs=[sampling_groth_stats]):
    # collect initial statistics
    stats = compute_graph_statistics(G)
    
    # run the confusion model 
    data = run_diffusion_model(G)
    
    return { "stats": stats,
            "analysis": analyse(G, analysis_funcs, data),
            "sim_data": data}
    

def analyse(G, analysis_funcs, sim_data):
    analysis_stats = {}
    # perform analysis functions
    if len(analysis_funcs) == 1:
        return analysis_funcs[0](G, sim_data)
    
    for func in analysis_funcs:
        analysis_stats[func.__name__] = func(G, sim_data)
        
    return analysis_stats

def run_graph_type_experiments(n_max, analysis_funcs=[sampling_groth_stats]):
    """Runs experiments on 3 type of networks for increasing number of nodes. Starts with number of nodes equal to the number of nodes in the treenetwork

    Args:

        n_final (int): final value of n
        analysis_funcs (list, optional): List with functions that analyse the data from "run_diffusion_model". Defaults to [sampling_groth_stats].

    Returns:
        dict: dictionary with all the results for the three tyoes of networks
    """
    # this is used to initialize the barabasi network forba_forest
    df = get_clean_dataset()
    B = generate_bipartite_network(df)
    forest_graph = tree_project_network(B)
    n_min = forest_graph.number_of_nodes()
    
    n_nodes = np.arange(n_min, n_max,step=2, dtype=int) # n *degree of random network must be even
    data = {}
    
    data["ba"] = {"experiments":[],"n":[]}
    data["ba_forest"] = {"experiments":[],"n":[]} 
    data["ra"] = {"experiments":[],"n":[]}
    
    for n in n_nodes:
        # experiments with normal barabasi network
        G = nx.barabasi_albert_graph(n, 2) # TODO the graph still needs diameter  and carbon properties
        G = generate_barabasi_forest(n_nodes=n, m=2)
        result = run_experiment(G, analysis_funcs)

        data["ba"]["experiments"].append(result)
        data["ba"]["n"].append(n)
        
        #expeeriments with barabasi network with starting point forest_graph
        G = nx.barabasi_albert_graph(n, 2, initial_graph=forest_graph)
        G = generate_barabasi_forest_from_forest(n_nodes=n, m=2, forest_graph=forest_graph)
        result = run_experiment(G, analysis_funcs)
        data["ba_forest"]["experiments"].append(result)
        data["ba_forest"]["n"].append(n)
        
        # experiments for random network
        G = generate_random_regular_graph(n_nodes=n, degree=6) # TODO: Help
        result = run_experiment(G, analysis_funcs)
        data["ra"]["experiments"].append(result)
        data["ra"]["n"].append(n)
        
    return data

def run_mutated_graph_experiment(G:nx.Graph, mutate_func, mod_args, analysis_funcs=[sampling_groth_stats]):
    """Mutates the graph and runs the diffusion model. We want to see what happens to the diffusion when we remove edges or nodes

    Args:
        G (nx.Graph): initial graph
        mutate_func (func): function that mutates the graph
        mod_args (list): list of args for the mod
        analysis_funcs (list, optional): functions that perform analysis . Defaults to [generic_analysis_func].

    Returns:
        dict: dictionary with network statsitics, diffusion analysis and simulation data.
    """
    
    mutate_func(G, mod_args)
    after_stats = compute_graph_statistics(G)
    sim_data = run_diffusion_model(G)
    
    analyse_stats = analyse(G, analysis_funcs, sim_data)
    
    return {
        "stats": after_stats,
        "analysis": analyse_stats,
        "sim_data": sim_data
    }