import pandas as pd
import networkx as nx
import numpy as np
from networkx.algorithms import bipartite


def get_clean_dataset(data_path="data/nph_3069_sm_tables2.xls"):
    """
    Get the cleaned (correct rows/cols, types, and column names) data from
    wood-wide web paper (https://nph.onlinelibrary.wiley.com/doi/full/10.1111/j.1469-8137.2009.03069.x)

    Keyword Args:
        data_path (string): Path to data set excel file

    Returns:
        pandas.DataFrame: data frame containing trees in the rows and fungal
            genets in the columns. A non-zero integer in the column B of row A
            means tree A is connected to genet B.
    """
    df = pd.read_excel(data_path, sheet_name="Sheet1",
                       header=5, skipfooter=5, usecols="A:B,D:P,R:AC") \
        .fillna(0) \
        .drop(0)   \
        .rename(columns={"Unnamed: 0": "Tree", "Unnamed: 1": "Cohort"})
    df = df.astype({k: int for k in df.columns[2:]})
    df["Tree"] = df["Tree"].replace("a|b", "", regex=True).astype(int)

    return df


def diameter_to_cohort(diameter):
    """
    Maps the diameter to the appropriate cohort.

    Args:
        diameter (float): diameter of the tree.

    Returns:
        cohort (string): this cohort is based on the diameter, signifying
        the age of the tree. The three cohorts are: Saplings, Maturing, and
        Established.

    """
    if diameter <= 15:
        cohort = "Sapling"

    elif diameter <= 35:
        cohort = "Maturing"

    else:
        cohort = "Established"

    return cohort


def generate_bipartite_network(df, carbon_scalar=500, stress_level=0):
    """
    Generate bipartite network with fungal and tree nodes as different
    bipartites.

    Args:
        df (pandas.DataFrame): Fungal network dataset

    Keyword Args:
        carbon_scaler (int): Diameter to carbon reserve conversion scalar
    """
    B = nx.Graph()

    # add genets nodes
    B.add_nodes_from(df.columns[2:], bipartite=0)

    cohort_diameter_map = {
        1: (0.7, 0.68),
        2: (8.1, 5.2),
        3: (24.5, 6.2),
        4: (46.4, 5.3)
    }

    def calc_carbon(diameter, max_diameter):
        fraction = diameter / max_diameter
        carbon = np.tanh((fraction - 0.5) * np.pi * 2) + stress_level
        return carbon

    # add tree nodes
    for _, row in df.iterrows():
        mean, stdev = cohort_diameter_map[row["Cohort"]]
        diameter = abs(np.random.normal(mean, stdev))
        B.add_node(
            row["Tree"],
            bipartite=1,
            cohort=diameter_to_cohort(diameter),
            diameter=diameter)

    max_diameter = max(nx.get_node_attributes(B, "diameter").values())

    for node in B.nodes():
        if B.nodes[node]["bipartite"] == 1:
            B.nodes[node]["carbon_value"] = (calc_carbon(
                B.nodes[node]["diameter"], max_diameter) + 1) * carbon_scalar

    # add edges between genets and trees
    edges = []
    for _, row in df.iterrows():
        for genet, n in row[2:].items():
            if n > 0:
                edges.append((genet, row["Tree"]))

    B.add_edges_from(edges)

    # remove disconnected nodes/islands
    B.remove_nodes_from([x for x in B.nodes() if B.degree(x) == 0])
    B.remove_nodes_from(("VES-11", 79))  # disconnected part

    return B


def tree_project_network(B) -> nx.Graph:
    """
    Project bipartite network to phytocentric network with trees as nodes and
    fungal connections as edges.

    Args:
        B (networkx.Graph): Bipartite network

    Returns:
        networkx.Graph: Tree-projected network
    """
    _, trees = bipartite.sets(B)
    return bipartite.weighted_projected_graph(B, trees)


def get_neighbors(G, i):
    """
    Get neighbor nodes of tree `i` in tree-projected network `G`

    Args:t
        G (networkx.Graph): Tree-projected network
        i (int): Tree node identifier

    Returns:l
        List[int]: Neighbors tree identifiers of node `i`
    """
    node_idx = list(G.nodes)[i]
    node = G.nodes[node_idx]

    return G.neighbors(node_idx)


def split3(xs, N):
    """
    Split xs into 3 equal N-sized lists. Used to work around `solve_ivp`'s
    limitation of one-dimensional state vectors to solve arbitrarily-sized
    multi-dimensional ODE's.

    Args:
        xs (List[float]): Flat list of state variables
        N (int): Number of nodes in network

    Returns:
        Tuple[List[float], List[float], List[float]]: Split state variables
    """
    return xs[:N], xs[N:2 * N], xs[2 * N:]


def diameter_growth(d, p, k, c, g):
    """
    Logistic function to represent diameter-dependent growth rate. 0 < c < 1
    should be chosen to represent a decay in growth rate as a tree ages
    (logistic decay).

    Returns:
        float: Carbon used for growth per time step for tree with diameter `d`
    """
    return g * 1 / (1 + c**(-d * k)) * p


def gaussian_uptake(d, A, μ, σ):
    """
    Gaussian function to represent diameter-dependent root carbon uptake.
    Smaller/young trees are more eager to take up carbon from their roots to use for
    growth and survival, while larger/older trees are less dependent on
    transferred carbon.

    Returns:
        float: Root carbon uptake per time step
    """
    return A * d * np.exp(-(d - μ)**2 / σ)


def diffusion_dynamics(t, y, G, D_C, N, uptake_ps, f, k, c, g, rho):
    """
    Diffusion-driven resource sharing ODE model of ectomycorrhizal networks in which
    trees are connected to common ectomycorrhizal networks to which they can
    transfer and obtain nutrients.

    Args:
        t (float): Time
        y (List[float]): State vector
        G (networkx.Graph): Tree-projected network
        D_C (float): Diffusion coefficient
        N (int): Number of nodes (trees) in network
        uptake_ps (Tuple[float]): Parameters for Gaussian uptake function
        f (float): Fraction of stored carbon transferred to root
        k (float): Conversion coefficient for diameter growth exponential
        c (float): Exponential base for diameter growth exponetnial (0 < c < 1)
        g (float): Conversion coefficient for growth term
        rho (float): Carbon to diameter conversion coefficient

    Returns:
        List[float]: Flat state vector of system at time `t`
    """
    nutrient_root, nutrient_plant, plant_diameter = split3(y, N)

    n_r = len(nutrient_root)
    n_p = len(nutrient_plant)
    n_d = len(plant_diameter)

    d_root = np.zeros(n_r)
    d_plant = np.zeros(n_p)
    d_diameter = np.zeros(n_d)

    for i, (r_i, p_i, d_i) in enumerate(
            zip(nutrient_root, nutrient_plant, plant_diameter)):
        # common terms between coupled DE's
        uptake = r_i * gaussian_uptake(d_i, *uptake_ps)
        deposition = f * p_i
        growth = diameter_growth(d_i, p_i, k=k, c=c, g=g)

        # change in root carbon, plant carbon, and diameter
        d_root[i] += -uptake + deposition
        d_plant[i] += uptake - deposition - growth
        d_diameter[i] += rho * growth

        # diffusion
        neighbors = get_neighbors(G, i)
        for j in neighbors:
            neighbor_node_idx = list(G.nodes).index(j)
            r_j = nutrient_root[neighbor_node_idx]

            d_root[i] += D_C * (r_j - r_i)

    return np.concatenate((d_root, d_plant, d_diameter))
