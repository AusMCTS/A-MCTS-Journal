"""
Information gathering task simulation using Cen-MCTS
Author: Nhat Nguyen (School of Computer Science - University of Adelaide)

"""


import sys; sys.path.append("../")
from envs.graph import Graph
from envs.graph_helper import import_oil_graph
from agent.Central_Agent import Central_InfoGathering_Agent
from datetime import datetime
from copy import deepcopy
import numpy as np
import argparse
import os
import csv


if __name__ == '__main__':

    # Parsing the input options.
    parser = argparse.ArgumentParser(description="Simulate the multi-agent tasks using Cen-MCTS")
    parser.add_argument("-s", "--save", help="Save performance", action='store_true', default=False)
    parser.add_argument("-f", "--folder", help="Folder name")
    parser.add_argument("-v", "--verbose", help="Print details", action='store_true', default=False)
    parser.add_argument("-p", "--params", help="Parameter testing", nargs="+", default=[])
    args = parser.parse_args()

    # System parameters.
    xL = 0                                           # min of the x-axis.
    xH = 200                                         # max of the x-axis.
    yL = 0                                           # min of the y-axis.
    yH = 100                                         # max of the y-axis.
    obsMask=np.array([[1,1,1,1,1,1,1,1],
                    [0,1,1,1,1,1,1,1],
                    [0,0,0,1,1,1,1,0],
                    [0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0]])
    agents = [100, 70]                               # agent starting position.

    # Fixed parameters.
    N_trial = 20                                      # number of trial experiments.
    reward_radius = 2.5                               # reward disk radius (km).
    c_p = 0.4                                         # Exploration parameter

    # Shared variational parameters.
    budget = 200                                      # km.
    planning_time = 60                                # (seconds).
    n_agents = 20                                     # number of agents.
    n_rewards = 200                                   # number of rewards.
    attrition_intensity = 0.5                         # proportion of agents fail.

    # Change parameters from keyboard.
    while len(args.params) > 0:
        param_name = args.params[0]
        param_value = args.params[1]
        del args.params[:2]
        if param_name == "agent":
            n_agents = int(param_value)
        elif param_name == "budget":
            budget = float(param_value)
        elif param_name == "planning":
            planning_time = float(param_value)
        elif param_name == "reward":
            n_rewards = int(param_value)
        elif param_name == "intensity":
            attrition_intensity = float(param_value)
        elif param_name == "comp":
            N_components = int(param_value)

    if args.save:
        if args.folder != None:
            directory = os.path.join("../../Data/", args.folder)
        else:
            now = datetime.now()
            directory = os.path.join("../../Data/", now.strftime("%Y-%m-%d-%H-%M"))
        if not os.path.isdir(directory):
            try:
                os.mkdir(directory)
            except:
                pass
        # Reward/rollout scores.
        scores = np.array(np.zeros([1, N_trial]))

    # Initialise a seed for reproductivity.
    rng =  np.random.default_rng(12345)

    # Parse the motion graph files.
    G = Graph(xL, xH, yL, yH, reward_radius, obsMask)
    G.add_node(agents[0], agents[1])
    G, locs = import_oil_graph(G, n_rewards)

    # Start simulation.
    for trial in range(N_trial):
        # Add rewards.
        rewards = rng.choice(locs, size=n_rewards, replace=False)
        G.reset_reward(n_rewards)
        G.add_reward(rewards)
        if args.verbose:
                print("Trial {}/{}".format(trial+1, N_trial))

        # Generate list of agents that will fail and the timestamp.
        attrition_idx = rng.choice(range(n_agents), size=int(n_agents*attrition_intensity), replace=False)

        # Create robots.
        initial_actions = []
        for j in range(len(G.edges_list)):
            if G.edges_list[j][0] == 0:
                initial_actions.append(j)

        actions_to_try = dict()
        for i in range(n_agents):
            actions_to_try[i] = deepcopy(initial_actions)

        robot = Central_InfoGathering_Agent(initial_state=np.nan, actions_to_try=actions_to_try, n_agents=n_agents, c_p=c_p, budget=budget, planning_time=planning_time, Z=deepcopy(G), n_rewards=n_rewards)

        # Centralised planning
        robot.planning()

        # Save values for analytics.
        final_paths = dict()
        for idx in range(n_agents):
            if idx not in attrition_idx:
                final_paths[idx] = robot.tree.data.at[0, 'best_rollout_path'][idx]
        score = sum(G.evaluate_traj_reward(final_paths))/len(rewards)
        if args.verbose:
            for path in final_paths.values():
                print(path)
            print(score)
        if args.save:
            scores[0][trial] = score
            np.savetxt("{}/Central-performance.csv".format(directory), scores, delimiter=",")
            with open("{}/Central-rollout-T{}.csv".format(directory, trial+1), "w", newline='') as f:
                write = csv.writer(f)
                for path in final_paths.values():
                    write.writerow([path])

