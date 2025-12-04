"""
Environment files constructor helper function.
Author: Nhat Nguyen (School of Computer Science - University of Adelaide)

"""

import sys; sys.path.append("../")
from envs.graph import Graph
from envs.prm import SampleFree, CollisionFree, Euclindean_dist, Near_radius
from matplotlib import pyplot as plt
from math import floor, ceil
from copy import deepcopy
import numpy as np
import csv
import argparse
import os


def generate_rewards(G:Graph, n_rewards:int):
    # Randomly generate the rewards initial locations, equally distributed.
    rewards = SampleFree(floor(n_rewards/4),
                        G.xL + 0 * (G.xH - G.xL),
                        G.xH - 0.5 * (G.xH - G.xL),
                        G.yL + 0 * (G.yH - G.yL),
                        G.yH - 0.5 * (G.yH - G.yL),
                        G, seed=np.random.randint(100))
    rewards = np.append(rewards, SampleFree(floor(n_rewards/4),
                        G.xL + 0 * (G.xH - G.xL),
                        G.xH - 0.5 * (G.xH - G.xL),
                        G.yL + 0.5 * (G.yH - G.yL),
                        G.yH - 0 * (G.yH - G.yL),
                        G, seed=np.random.randint(100)), axis=0)
    rewards = np.append(rewards, SampleFree(ceil(n_rewards/4),
                        G.xL + 0.5 * (G.xH - G.xL),
                        G.xH - 0 * (G.xH - G.xL),
                        G.yL + 0 * (G.yH - G.yL),
                        G.yH - 0.5 * (G.yH - G.yL),
                        G, seed=np.random.randint(100)), axis=0)
    rewards = np.append(rewards, SampleFree(ceil(n_rewards/4),
                        G.xL + 0.5 * (G.xH - G.xL),
                        G.xH - 0 * (G.xH - G.xL),
                        G.yL + 0.5 * (G.yH - G.yL),
                        G.yH - 0 * (G.yH - G.yL),
                        G, seed=np.random.randint(100)), axis=0)

    return rewards


def generate_new_graph(G:Graph, n_agents:int, n_nodes:int, n_rewards:int, path:str):
    # Initialised the agents positions.
    agents = SampleFree(floor(n_agents),
                        G.xL,
                        G.xH,
                        G.yL,
                        G.yH,
                        G, seed=np.random.randint(100))
    np.savetxt("{}/agents.csv".format(path), agents, delimiter=",")
    for i in range(n_agents):
        G.add_node(agents[i][0], agents[i][1])

    # Generate the intermediate nodes.
    nodes = np.zeros((0, 2))
    x = G.xL
    while x <= G.xH:
        nodes = np.append(nodes, np.array([[x, G.yL], [x, G.yH]]), axis=0)
        x += max_edge_length
    x = G.yL
    while x <= G.yH:
        nodes = np.append(nodes, np.array([[G.xL, x], [G.xH, x]]), axis=0)
        x += max_edge_length

    nodes = np.append(nodes, SampleFree(n_nodes-len(nodes),
                        G.xL,
                        G.xH,
                        G.yL,
                        G.yH,
                        G, seed=np.random.randint(100)), axis=0)

    np.savetxt("{}/nodes.csv".format(path), nodes, delimiter=",")
    for i in range(len(nodes)):
        G.add_node(nodes[i][0], nodes[i][1])

    # Generate rewards.
    rewards = generate_rewards(G, n_rewards)
    np.savetxt("{}/rewards.csv".format(path), rewards, delimiter=",")
    for i in range(len(rewards)):
        G.add_node(rewards[i][0], rewards[i][1])

    # Generate edges between nodes.
    for v in G.nodes.items():
        U = Near_radius(G, v[1], r=max_edge_length)
        for u in U:
            if CollisionFree(G, v[1], u[1]):
                G.add_edge(v[0], u[0], Euclindean_dist(u[1], v[1]), n_rewards)
    with open("{}/edges.csv".format(path), 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for start in G.edges.keys():
            for end in G.edges[start].keys():
                writer.writerow([start, end,])

    # Check what edge covers what rewards.
    G.add_reward(rewards)
    return G, agents, rewards, nodes


def import_graph(G:Graph, dir:str):
    # Get existing agents positions.
    agents = np.genfromtxt("{}/agents.csv".format(dir), delimiter=",")
    n_agents = len(agents)
    if n_agents > 2:
        for i in range(n_agents):
            G.add_node(agents[i][0], agents[i][1])
    elif n_agents == 2:
        x = agents[0]
        y = agents[1]
        G.add_node(x, y)
        n_agents = 1
        agents = np.array([[x, y]])
    else:
        print("Format Error!")
        sys.exit()

    # Get existing nodes latlongs.
    nodes = np.genfromtxt("{}/nodes.csv".format(dir), delimiter=",")
    n_nodes = len(nodes)
    for i in range(len(nodes)):
        G.add_node(nodes[i][0], nodes[i][1])

    # Get existing rewards.
    rewards = np.genfromtxt("{}/rewards.csv".format(dir), delimiter=",")
    n_rewards = len(rewards)
    for i in range(len(rewards)):
        G.add_node(rewards[i][0], rewards[i][1])

    # Generate edges.
    with open("{}/edges.csv".format(dir), 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        for edges in reader:
            if edges:
                start = int(edges[0])
                end = int(edges[1])
                G.add_edge(start, end, Euclindean_dist(G.find_coordinates(start), G.find_coordinates(end)), n_rewards)

    return G, agents, rewards, nodes, n_agents, n_nodes, n_rewards


# Modify the number of rewards.
def modify_number_rewards(G: Graph, n_rewards:int, rewards, seed=0):
    # Delta between the desired number of rewards and the current.
    nNew = int(n_rewards) - rewards.shape[0]
    # Add more rewards.
    if nNew > 0:
        rewards = np.append(rewards, SampleFree(nNew, G.xL, G.xH, G.yL, G.yH, G, seed), axis=0)
    # Remove some rewards.
    else:
        rng = np.random.default_rng(seed)
        rewards = rng.choice(rewards, n_rewards, replace=False)
    return rewards


# Parsing the rollout files.
def parse(d):
    dictionary = dict()
    # Removes curly braces and splits the pairs into a list
    pairs = d.strip('{}').split('], ')
    for i in pairs:
        pair = i.split(': ')
        x = pair[1].strip('[]').split(', ')
        if x[0] != '':
            dictionary[int(pair[0])] = [float(num) for num in x[0:]]
    return dictionary


def draw_graph(G:Graph, agents, rewards):
    G.draw_edges()
    G.draw_nodes()
    G.draw_agents(agents)
    G.draw_rewards(rewards)
    G.draw_obs()
    plt.show()

def import_oil_graph(G:Graph, n_rewards:int):
    # Add nodes.
    nodes = np.genfromtxt("../../Data/config/nodes.csv", delimiter=",")
    n_nodes = len(nodes)
    for i in range(n_nodes):
        G.add_node(nodes[i][0], nodes[i][1])

    # Add edges.
    with open("../../Data/config/edges.csv", 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        for edges in reader:
            if edges:
                start = int(edges[0])
                end = int(edges[1])
                G.add_edge(start, end, Euclindean_dist(G.find_coordinates(start), G.find_coordinates(end)), n_rewards)

    # Read oil rigs locations.
    locs = np.genfromtxt("../../Data/config/locs.csv", delimiter=",")
    return G, locs


if __name__ == '__main__':
    # Parsing the input options.
    parser = argparse.ArgumentParser(description="Graph constructor")
    parser.add_argument("-a", "--animation", help="Show constructed graph", action='store_true', default=False)
    parser.add_argument("-n", "--n_configs", help="No of configurations", default=1)
    parser.add_argument("-d", "--draw", help="Draw existing graph")
    args = parser.parse_args()

    # Declare system size.
    xL = -2     # min of the x-axis
    xH = 2      # max of the x-axis
    yL = -2     # min of the y-axis
    yH = 2      # max of the y-axis
    obsMask = np.array([
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ])                                  # the obstacles mask.

    # Declare graph configurations.
    n_rewards = 200                             # number of reward disks.
    n_nodes = 200                               # number of intermediate nodes.
    n_agents = 10                               # number of agents.
    reward_radius = 0.05                        # reward disk radius.
    max_edge_length = 0.8                       # max edge length between any 2 nodes.
    n_configs = int(args.n_configs)             # Number of configurations to construct.

    # Parent Directory path to save the files.
    parent_dir = "../../Data/"

    if args.draw:
        dir = "../../Data/{}".format(args.draw)
        G = Graph(xL, xH, yL, yH, reward_radius, obsMask)
        G, agents, rewards, _, _, _, _ = import_graph(G, dir)
        draw_graph(G, agents, rewards)

    else:
        for i in range(n_configs):
            # Create config directory.
            directory = "Config_{}".format(i)
            path = os.path.join(parent_dir, directory)
            if not os.path.isdir(path):
                os.mkdir(path)
            # Create graph.
            G = Graph(xL, xH, yL, yH, reward_radius, obsMask)
            G, agents, rewards, nodes = generate_new_graph(G, n_agents, n_nodes, n_rewards, path)

        if args.animation and n_configs == 1:
            draw_graph(G, agents, rewards)

