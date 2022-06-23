import imp
from matplotlib import pyplot as plt
import networkx as nx 
from networkx.algorithms import bipartite
import numpy as np
import collections

def cumsum_degree_distr(G, bipartite_set):
    degrees = sorted([G.degree(n) for n in bipartite_set], reverse=True)
    degreeCount = collections.Counter(degrees)
    deg, cnt = zip(*degreeCount.items())
    cs = np.cumsum(cnt)
    return deg, cs

def plot_bipartite_cumdegree_dist(G, trees, fungi, ax=None):
    
    tree_deg, tree_cs = cumsum_degree_distr(G, trees)
    fungi_deg, fungi_cs = cumsum_degree_distr(G, fungi)
    
    if ax== None:
        fig, ax = plt.subplots()
        
    ax.loglog(tree_deg, tree_cs, 'r--o', label="trees")
    ax.loglog(fungi_deg, fungi_cs, 'b--o', label="fungae")
    ax.grid(True, which="both")
    ax.set(title="Cumulative degree distribution of bipartite network",
           xlabel="k",
           ylabel="P(k)")
    plt.legend()
    plt.show()
    
