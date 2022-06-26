# Ectomycorrhizal Network Dynamics Model
This repository contains the code for the simulation and visualization of an
ODE- and network-based model of tree-to-tree nutrient transfer through
ectomycorrhizal networks (EMNs). The model's dynamics are based on the source-sink
hypothesis, where older/bigger trees act as sources of carbon for
smaller/younger trees (seedlings). The model is largely based on the data and
results from "Architecture of the wood-wide web: _Rhizopogon_ spp. genets link multiple Douglas-fir cohorts" (https://doi.org/10.1111/j.1469-8137.2009.03069.x).
The model dynamics are greatly simplified and do not fully capture the
complex and diverse dynamics in real-world EMNs. However, this simple model
displays directed carbon (C) transfer through common EMNs favoring seedlings
that have higher respiration compared to older trees. This in turn leads to
competition between seedlings, where some seedlings outcompete others due to
being connected to better or more trees. These dynamics emerge from simple,
local node-to-node rules.
