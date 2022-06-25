from matplotlib.cm import get_cmap
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pandas import read_excel

def parse_forestdata_to_graph():
    df = read_excel("data/nph_3069_sm_tables2.xls", sheet_name="Sheet1",
                   header=5, skipfooter=5, usecols="A:B,D:P,R:AC") \
        .fillna(0) \
        .drop(0)   \
        .rename(columns={"Unnamed: 0": "Tree", "Unnamed: 1": "Cohort"})
    
    df = df.astype({k: int for k in df.columns[2:]})
    df["Tree"] = df["Tree"].replace("a|b", "", regex=True).astype(int)
    df
        #Initialize graph
    B = nx.Graph()

    # add genets nodes
    B.add_nodes_from(df.columns[2:], bipartite=0)

    #Cohort-diameter map
    cohort_diameter_map = {1: (0.7, 0.68), 2: (8.1, 5.2), 3: (24.5, 6.2), 4: (46.4, 5.3)}

    def calc_carbon(diameter, max_diameter, stress_level=0):
        fraction = diameter/max_diameter
        carbon = np.tanh((fraction - 0.5)*np.pi*2) + stress_level
        return carbon
        
    # add tree nodes
    for _, row in df.iterrows():
        mean, stdev = cohort_diameter_map[row["Cohort"]]
        B.add_node(row["Tree"], bipartite=1, cohort=row["Cohort"], diameter=abs(np.random.normal(mean, stdev)))

    max_diameter = max(nx.get_node_attributes(B, "diameter").values())    

    for node in B.nodes():
        if (B.nodes[node]["bipartite"] == 1):
            B.nodes[node]["carbon_value"] = (calc_carbon(B.nodes[node]["diameter"], max_diameter) + 1)*5

    edges = []
    # add edges between genets and trees
    for _, row in df.iterrows():
        for genet, n in row[2:].items():
            if n > 0:
                edges.append((genet, row["Tree"]))
                
    B.add_edges_from(edges)
    B.remove_nodes_from([x for x in B.nodes() if B.degree(x) == 0])
    B.remove_nodes_from(("VES-11", 79))
    
    return B

def get_forest_projected_tree_network():
    F = parse_forestdata_to_graph()
    
    trees, fungi = nx.bipartite.sets(F)
    G = nx.bipartite.weighted_projected_graph(F, trees)
    
    return G
    
    

def draw_biparite_network(G, trees):
    pos = nx.bipartite_layout(G, trees, align="horizontal")

    node_sizes = []
    node_colors = []
    for node in G.nodes:
        if G.nodes[node]["bipartite"] == 1:
            node_sizes.append(60)
            node_colors.append("green")
        else:
            node_sizes.append(225)
            if node[0:3] == "VES":
                node_colors.append("#5e4125")
            else:
                node_colors.append("#877a2d")

    plt.figure(figsize=(8, 3.5), dpi=300, facecolor="w", frameon=False)
    nx.draw_networkx_nodes(G, pos=pos, node_size=node_sizes, node_color=node_colors, linewidths=0.3, edgecolors="white")
    nx.draw_networkx_labels(G, pos, font_color="white", font_size=4.15)
    nx.draw_networkx_edges(G, pos, alpha=0.55, width=[0.5 for _ in range(len(G.nodes))])
    plt.gca().set_axis_off()
    plt.show()

def visualize_carbon_network(G, pos=None):
    if not pos:
        pos = nx.spring_layout(G, k=3,seed=99192, weight=None)
        
    node_sizes = []

    for node in G.nodes:
        node_size = max(10, G.nodes[node]["diameter"]*6)
        node_sizes.append(node_size)
        
    degrees_dict = {n: d for n, d in G.degree()}
    degrees = list(degrees_dict.values())
    carbon_values = list(nx.get_node_attributes(G, "carbon_value").values())
    sorted_degrees = sorted(degrees_dict, key=degrees_dict.get)

    plt.figure(figsize=(5, 4), dpi=300, facecolor="w", frameon=False)
    nx.draw_networkx_nodes(G, pos=pos, node_size=node_sizes, node_color=carbon_values,
                           edgecolors="white", linewidths=0.3, cmap=get_cmap("coolwarm"), vmin=0, vmax=10)
    #nx.draw_networkx_labels(G, pos, labels = nx.get_node_attributes(G, "carbon_value"), font_color="black", font_size=4.1)
    nx.draw_networkx_edges(G, pos, alpha=0.4, width=[0.2 for _ in range(len(G.nodes))])
    plt.gca().set_axis_off()
    plt.show()