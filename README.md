# Ectomycorrhizal Network Dynamics Model
This repository contains the code for the simulation and visualization of an
ODE- and network-based model of tree-to-tree nutrient transfer through
ectomycorrhizal networks (EMNs). The model's dynamics are based on the source-sink
hypothesis, where older/bigger trees act as sources of carbon for
smaller/younger trees (saplings). The model is largely based on the data and
results from "Architecture of the wood-wide web: _Rhizopogon_ spp. genets link multiple Douglas-fir cohorts" (https://doi.org/10.1111/j.1469-8137.2009.03069.x).
The model dynamics are greatly simplified and do not fully capture the
complex and diverse dynamics in real-world EMNs. However, this simple model
displays directed carbon (C) transfer through common EMNs favoring saplings
that have higher respiration demands compared to older trees. This in turn leads to
competition between saplings, where some saplings outcompete others due to
being connected to better or more trees. These dynamics emerge from simple,
local node-to-node rules.



https://user-images.githubusercontent.com/7383594/176503913-77a2755c-dbb1-4ee9-a557-ca55adb386ea.mp4

## Repository Structure
- `data/`: Contains network data from paper used in experiments and saved data
  from computationally demanding experiments.
- `figures/`: Plots and network visualizations used in presentation.
- `movie_frames/`: Contains animation(s) of network.
- `src/`: Python source files used in experiments/simulations.

Notebooks containing visualization and analysis code for various experiments are
found in the root of the repository. The notebooks contain the following
experiments:
- `Forest Data Experiments.ipynb`: Main results and experiments on the forest
  data from the Beiler paper.
- `Forest Attack and Failure Experiment.ipynb`: Attack and failure experiments
  on forest data.
- `Average Degree vs. Average Growth Experiment.ipynb`: Experiment on the
  importance of connectedness on growth.
- `Characteristic Time Experiment.ipynb`: Experiment on the influence of
  connectedness on the characteristic time.
- `Network Dynamics Animation.ipynb`: Notebook for creating the animation of
  carbon spread and growth throughout the forest network.
- `Barabasi-Albert Network Experiments.ipynb`: Experiments repeated on generated
  Barbasi-Albert scale-free network.
- `Erdos-Renyi Network Experiments.ipynb`: Experiments repeated on generated
  Erdos-Renyi random network.
