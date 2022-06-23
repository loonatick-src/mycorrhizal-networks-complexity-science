
#%%
from copy import copy, deepcopy
from typing import Any
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from tqdm import tqdm, trange
import random
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community.quality import modularity
from networkx.algorithms.cluster import average_clustering
import numpy as np
import matplotlib.pyplot as plt
#%%

random.seed(900)

class Tree(Agent):
    def __init__(self, unique_id: int, model: Model, graph: nx.Graph, size) -> None:
        super().__init__(unique_id, model)
        self.size = size
        self.reserve = 1
        self.graph = graph
        self.model
        
    @property
    def neighbors_ids(self):
        return self.graph.neighbors(self.unique_id)

    def step(self):
        self.produce()
        self.consume()
        self.grow()
        
        # check if tree has enough reserve to share
        
    def produce(self):
        self.reserve += self.size * self.model.sunlight
        
     
    def consume(self):
        self.reserve-=1

    def grow(self): 
        if self.reserve > self.model.threshold:
            self.reserve-=1
            self.size+=1


class ForestModel(Model):
    def __init__(self,nodes, m, sunlight=5, 
                 initial_reserve=20,
                 growthreshold=10, share_threshold=10,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.graph:nx.Graph = nx.barabasi_albert_graph(nodes, m)
        self.grow_threshold = growthreshold
        self.share_threshold = share_threshold
        
        # if the sunlight is less then 1 the size has to be greater then one to not run out of resources
        self.sunlight = sunlight
        
        for node in self.graph.nodes():
            self.graph.nodes[node]["size"] = 10 * random.random()
            self.graph.nodes[node]["reserve"] = initial_reserve
            # self.graph[node][""]
            
    def step(self):
        #individual steps
        nodes = deepcopy(self.graph.nodes())
        # self.graph.nodes()
        for node in nodes:
            self.node_step(node)
        
        # share resources
        rich_nodes = [node for node in self.graph.nodes(data=True) 
                     if node[1]["reserve"] > self.share_threshold]
        rich_nodes_ids = [node[0] for node in rich_nodes]
        
        
        for node in rich_nodes_ids:
            neighbors = self.graph.neighbors(node)
            pour_neighbors = [n for n in neighbors if n not in rich_nodes_ids]
            
            n_nbrs = len(pour_neighbors)
            share = 0
            if n_nbrs > 0:
                share = 1 / n_nbrs
            self.graph.nodes[node]["reserve"] -= 1
            for node in pour_neighbors:
                self.graph.nodes[node]['reserve'] += share
                
    def node_step(self, node):
        # produce
        self.graph.nodes[node]["reserve"] += self.sunlight * self.graph.nodes[node]["size"]
        
        # consume 
        self.graph.nodes[node]["reserve"] -= self.graph.nodes[node]["size"] / self.graph.nodes[node]["reserve"] +1 
        if self.graph.nodes[node]["reserve"] < 0:
            # die mf die
            self.graph.remove_node(node)
            return
        
        # grow 
        if self.graph.nodes[node]["reserve"] > self.grow_threshold:
            self.graph.nodes[node]["reserve"] -= 1
            if self.graph.nodes[node]['size'] < 50:
                self.graph.nodes[node]["size"] += 1
            
            
    def collect(self, data):
        nodes = self.graph.nodes(data=True)
        total_trees = len(nodes)
        total_reserve = 0
        total_sharing = 0
        total_growing = 0
        for node in nodes:
            total_reserve += node[1]["reserve"]
            if node[1]["reserve"] > self.grow_threshold: total_growing+=1
            if node[1]["reserve"] > self.share_threshold: total_sharing+=1
            
        data["total_trees"].append(total_trees)
        data["total_reserve"].append(total_reserve)
        data["total_growing"].append(total_growing)
        data["total_sharing"].append(total_sharing)

data = {
    "total_reserve": [],
    "total_trees": [],
    "total_sharing": [],
    "total_growing": [],
}

steps=140
time_arr = np.arange(steps)
model = ForestModel(100, m=2, sunlight=0.2, initial_reserve=11, growthreshold=10, share_threshold=20)

for i in range(steps): 
    model.step()
    model.collect(data)
    pos = nx.spring_layout(model.graph)
    # fig, ax = plt.subplots()
    # nx.draw_networkx_nodes(model.graph, pos=pos, #node_size=node_sizes, node_color=carbon_vals_t,
    #                    edgecolors="black", linewidths=0.3, cmap="coolwarm",ax=ax, vmin=-1, vmax=1)
    

#%%
fig, ax = plt.subplots(4,1, figsize=(10,10))
for i, (key,value) in enumerate(data.items()):
    ax[i].plot(time_arr, value)
    ax[i].set(title=key)

# ax[0].plot(time_arr, data["total_growing"])
# ax[1].plot(time_arr, data["total_reserve"])
# ax[2].plot(time_arr, data["total_trees"])
# ax[3].plot(time_arr, data["total_sharing"])

# %%
