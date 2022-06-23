import imp
from matplotlib import pyplot as plt
import networkx as nx 
from networkx.algorithms import bipartite
import numpy as np
import collections
from scipy.special import zeta 
from scipy.stats import kstest

def get_degrees_sorted(G, bipartite_set):
    return sorted([G.degree(n) for n in bipartite_set], reverse=True)

def cumsum_degree_distr(G, bipartite_set):
    degrees = get_degrees_sorted(G, bipartite_set)
    degreeCount = collections.Counter(degrees)
    deg, cnt = zip(*degreeCount.items())
    cs = np.cumsum(cnt)
    return deg, cs

def plot_bipartite_cumdegree_dist(G, trees, fungi, ax=None, plot_powerlaw=True):
    
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
    # if plot_powerlaw:
    #    tree_degree_approx = approx_powerlaw(G, trees) 
    #    fungi_degree_approx = approx_powerlaw(G, fungi) 
    
    plt.legend()
    plt.show()
    
def approx_powerlaw(G, bipartite_set):
    """This function should approximate the powerlaw of the cumulative distribution using the optimal kmin value """
    return 0
    # TODO 
    
def gamma(Ks, kmin): 
    """ $$ \gamma  = 1 + N\left[ {\sum\limits_{i = 1}^N {\ln \frac{{k_i }}{{K_{\min }  - \frac{1}{2}}}} } \right]^{ - 1}  \hspace{20 mmr$$ """
    # print(Ks)
    summ = sum([np.log(ki/(kmin - 1/2)) for ki in Ks])
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