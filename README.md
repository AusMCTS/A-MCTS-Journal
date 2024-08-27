# A-MCTS

Decentralized Multi-Agent Planning With Attrition (IEEE Transactions on Mobile Computing)

This project contains the source code for the environment constructor, planning algorithms, and simulations used for the IEEE Transactions on Mobile Computing paper listed above, implemented in the Python programming language. The repository contains several folders:

- `algos`: generic MCTS implementation,
- `envs`: graph constructor using PRM algorithm,
- `agent`: generic agent classes implementation, can be extended to other applications:
    - `agents/Base_Agent`: abstract class implementation for decentralised MCTS,
    - `agents/DecMCTS_Agent`, `agents/SwMCTS_Agent`, `agents/AMCTS_Agent`: abstract classes implementation for Dec-MCTS, SW-MCTS, and A-MCTS,
    - `agents/InfoGathering_Agent`: agent classes for information gathering domain,
    - `agents/Attrition_Agent`: base agent class under attrition risks.

## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements libraries.
```bash
pip install -r requirements.txt
```

Or use the [conda](https://docs.conda.io/projects/conda/en/stable/) to create a testing environment.
```bash
conda create --name <env> --file requirements.txt
```

## Usage
To run a simulation of a multi-agent path planning task.
```bash
python3 simulation.py [-h] [-s] -m {Dec, A, Global, Greedy, Reset} [-f FOLDER] [-v] [-p PARAMS [PARAMS ...]]
optional arguments:
  -h, --help            show this help message and exit
  -s, --save            Save performance
  -m {Dec, A, Global, Greedy, Reset}, --mode {Dec, A, Global, Greedy, Reset}
                        Algorithm mode
  -f FOLDER, --folder FOLDER
                        Folder name to store simulation data
  -v, --verbose         Print details
  -p PARAMS [PARAMS ...], --params PARAMS [PARAMS ...]
                        Parameter testing

```

To construct new environment configurations.
```bash
python3 graph_helper.py [-h] [-a] [-n N_CONFIGS] [-d DRAW]

Graph constructor

optional arguments:
  -h, --help            show this help message and exit
  -a, --animation       Show constructed graph
  -n N_CONFIGS, --n_configs N_CONFIGS
                        No of configurations
  -d DRAW, --draw DRAW  Draw existing graph
```
