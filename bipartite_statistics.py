import imp
from matplotlib import pyplot as plt
import networkx as nx 
from networkx.algorithms import bipartite
import numpy as np
import collections
from scipy.special import zeta 
from scipy.stats import kstest

def get_degrees_sorted(G, bipartite_set, descending=True):
    return sorted([G.degree(n) for n in bipartite_set], reverse=descending)

def cumsum_degree_distr(degrees):
    degreeCount = collections.Counter(degrees)
    deg, cnt = zip(*degreeCount.items())
    cs = np.cumsum(cnt)
    return deg, cs

def plot_bipartite_cumdegree_dist(G, trees, fungi, ax=None, plot_powerlaw=True):
    from seaborn import set_style, despine
    
    set_style("whitegrid")
    tree_degrees = get_degrees_sorted(G, trees)
    fungi_degrees = get_degrees_sorted(G, fungi)

    cum_tree_deg, tree_cs = cumsum_degree_distr(tree_degrees)
    cum_fungi_deg, fungi_cs = cumsum_degree_distr(fungi_degrees)
    
    if ax== None:
        fig, ax = plt.subplots()
    despine(fig=fig, ax=ax)        
    ax.loglog(cum_tree_deg, tree_cs/sum(tree_degrees), 'r--o', label="trees")
    ax.loglog(cum_fungi_deg, fungi_cs/sum(fungi_degrees), 'b--o', label="fungae")
    ax.grid(True, which="both")
    ax.set(title="Cumulative degree distribution of bipartite network",
           xlabel="k",
           ylabel="P(k)")
    
    # Sorry guys, i did not manage to do this...    
    # if plot_powerlaw:
    #     ax.loglog(cum_tree_deg, powerlaw(cum_tree_deg, gamma(tree_degrees, 3)),'r--')
    #     ax.loglog(cum_fungi_deg, powerlaw(cum_fungi_deg, gamma(fungi_degrees, 3)),'b--')
        # ax.loglog(cum_tree_deg, tree_degree_approx, 'r--', label="trees")
        # ax.loglog(cum_fungi_deg, fungi_degree_approx, 'b--', label="fungae")
    
    plt.legend()
    plt.show()
 
def powerlaw(x, gamm):
    return gamm*np.array(x**(-gamm))
    
def approx_powerlaw(G, bipartite_set, degrees):
    """This function should approximate the powerlaw of the cumulative distribution using the optimal kmin value """
    kmin, kmin_index = find_kmin(degrees)
    gamm = gamma(degrees, kmin)
    cumsum_powerlaw = Pk(degrees, gamm, kmin)
    return  cumsum_powerlaw
    # TODO 
    
def gamma(Ks, kmin_idx=0): 
    """ $$ \gamma  = 1 + N\left[ {\sum\limits_{i = 1}^N {\ln \frac{{k_i }}{{K_{\min }  - \frac{1}{2}}}} } \right]^{ - 1}  \hspace{20 mmr$$ """
    # print(Ks)
    kmin = Ks[kmin_idx]
    summ = sum([np.log(ki/(kmin - 1/2)) for ki in Ks[kmin_idx:]])
    # print (summ)
    return (1 + len(Ks) * (summ**(-1)))

def pk (Ks, gamma, kmin): 
    """degree distribution with Ks is the list of degrees """
    return 1/(zeta(gamma, kmin)*Ks**(-gamma))

def Pk(Ks, gamma, kmin): 
    """ cumulative degree distribution function (CDF) where Ks is a list of degrees """
    return 1 - zeta(gamma, Ks) / zeta(gamma, kmin)
    
def maximum_distance(Ks, kmin):
    ks = [k for k in Ks if k>=kmin]
    gamm = gamma(Ks, kmin)
    cdf = Pk(Ks, gamm, kmin)
    stat, p_value = kstest(ks, cdf)
    return np.abs(stat), p_value

def find_kmin(Ks:list):
    """Finds the kmin with the best fit of powerlaw """
    D_list = []
    for i, kmin in enumerate(Ks[:]):
        D, _ = maximum_distance(Ks[i:], kmin)
        D_list.append(D)
        
    optimal_kmin_idx = np.argmin(D_list)
    return Ks[optimal_kmin_idx], optimal_kmin_idx

def average_degree_partite(G, fungi):
    """ returns the average degree of both parties 

    Args:
        G (nx.Graph): graph
        fungi (set): the later set

    Returns:
        tuple: average degree of the party 0 and 1
    """
    ks_trees, ks_fungi = bipartite.degrees(G, fungi)

    bipartite_avg_degree = lambda  ks: np.sum([k[1] for k in ks])/len(ks)
    avg_k_trees = bipartite_avg_degree(ks_trees)
    avg_k_fungi = bipartite_avg_degree(ks_fungi)
    return avg_k_trees, avg_k_fungi