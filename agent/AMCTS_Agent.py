"""
A-MCTS agent implementations.
Author: Nhat Nguyen (School of Computer Science - University of Adelaide)

"""


from agent.Base_Agent import TreeExhaustedException
from agent.DecMCTS_Agent import DecMCTS_Agent
from algos.mcts import growTree
from algos.functions import random_choice_bias
from copy import deepcopy
import numpy as np
import pandas as pd
import logging
import time
import shared_info


class AMCTS_Agent(DecMCTS_Agent):
    '''Abstract A-MCTS agent implementations.'''
    def __init__(self, initial_state:float, initial_actions:list, initial_position:list, i:int, n_agents:int, gamma:float, c_p:float, budget:float, planning_time:float, N_components:int, N_com_every:int, RM_iter:int, msg_threshold:int=0, logger:logging=None):
        super().__init__(initial_state, initial_actions,initial_position, i, n_agents, gamma, c_p, budget, planning_time, N_components, N_com_every, logger)
        # Iteration budget to run RM.
        self.RM_iter = RM_iter
        # Allowed missed message.
        self.msg_threshold = msg_threshold
        # Keep track of other agents status.
        self.status = dict()
        self.miss_msg = dict()
        for i in range(n_agents):
            if i != self.index:
                self.miss_msg[i] = 0

    def check_status(self):
        '''Remove agent info if it has not communicated in a while.'''
        flag = False
        idx_to_remove = list()
        for i in range(self.n_agents):
            if i != self.index and not np.isinf(self.status[i]):
                if self.distribution[i].iter[0] == self.status[i]:
                    self.miss_msg[i] += 1
                    if  self.miss_msg[i] > self.msg_threshold:
                        idx_to_remove.append(i)
                        self.miss_msg[i] = 0
                        self.distribution[i] = pd.DataFrame(data={'prob': [1], 'path': [list()], 'tree_idx': [0], 'iter': [-np.inf]})
                        flag = True
                else:
                    self.miss_msg[i] = 0
        if flag:
            # Clear info in global distribution.
            with shared_info.global_distribution_lock:
                for i in idx_to_remove:
                    shared_info.global_distribution[self.index][i] = pd.DataFrame(data={'prob': [1], 'path': [list()], 'tree_idx': [0], 'iter': [-np.inf]})
        return flag

    def search_best_response_policy(self):
        '''Search for best response joint policies using regret matching.'''
        # Set up the cumlative regret dict for each agent
        active_robots = list()
        for index in self.distribution.keys():
            if len(self.distribution[index].path[0]) > 0:
                active_robots.append(index)
        cumulative_regrets = dict()
        for idx in active_robots:
            cumulative_regrets[idx] = np.zeros(len(self.distribution[idx].prob))

        # Best global payoff.
        best_payoff = 0

        # Compute the strategies using regret maching
        for _ in range(self.RM_iter):
            action_sequences = dict()
            # Sample action sequences for every agents
            for index in active_robots:
                action_sequences[index] = random_choice_bias(self.distribution[index]['path'].copy().tolist(),
                                                        self.distribution[index]['prob'].copy().tolist())
            # Calculate the global payoff
            actual_payoff = self.f_reward(action_sequences)
            if actual_payoff > best_payoff:
                best_payoff = actual_payoff
                self.best_response = deepcopy(action_sequences)

            # Update the cumulative regrets for each action sequence of each robot
            for index in active_robots:
                # Copy values for calculation
                what_if_actions = deepcopy(action_sequences)
                other_actions = self.distribution[index]['path'].copy().tolist()
                # Regrets equal difference between what-if action and actual action
                for i in range(len(other_actions)):
                    what_if_actions.update({index: other_actions[i]})
                    cumulative_regrets[index][i] += self.f_reward(what_if_actions) - actual_payoff
                # Update regret-matching strategy
                pos_cumulative_regrets = np.maximum(0, cumulative_regrets[index])
                if sum(pos_cumulative_regrets) > 0:
                    self.distribution[index].prob = pos_cumulative_regrets / sum(pos_cumulative_regrets)
                else:
                    self.distribution[index].prob = np.full(shape=len(self.distribution[index].prob), fill_value=1/len(self.distribution[index].prob))

    def planning(self):
        '''Planning method.'''
        # Planning start.
        # if self.logger:
        #     self.logger.info("Agent {} planning...".format(self.index))
        start = time.time()
        while (time.time() - start) < self.planning_time:
            for _ in range(self.N_com_every):
                # Rollout some simulations.
                for _ in range(10):
                    self.iter += 1
                    # Re-initialise backprop.
                    f_backprop = lambda tree, tree_idx, current_score, current_rollout: self.f_backprop(tree, tree_idx, current_score, current_rollout)
                    # Grow the tree search.
                    _, rollout_path = growTree(self.tree, self.root_idx, self.f_score, self.f_actions, self.f_terminal, self.f_ucb, self.f_sampler, f_backprop)
                    self.rollout_history.append(rollout_path)

            # Compress the tree into product distribution.
            try:
                self.compress_tree(self.N_components)
            except TreeExhaustedException:
                self.communicate()
                break
            # Share with others.
            for i in range(self.n_agents):
                self.status[i] = self.distribution[i].iter[0]
            self.communicate()
            # Update set of active agents.
            self.best_response.clear()
            _ = self.check_status()
            # Find the join policy.
            self.search_best_response_policy()


class GreedyMCTS_Agent(AMCTS_Agent):
    '''Abstract Greedy-MCTS agent implementations.'''
    def __init__(self, initial_state: float, initial_actions: list, initial_position: list, i: int, n_agents: int, gamma: float, c_p: float, budget: float, planning_time: float, N_components: int, N_com_every: int, RM_iter: int, msg_threshold: int = 0, logger: logging = None):
        super().__init__(initial_state, initial_actions, initial_position, i, n_agents, gamma, c_p, budget, planning_time, N_components, N_com_every, RM_iter, msg_threshold, logger)

    def search_best_response_policy(self):
        '''Search for best response joint policies using sequential greedy.'''
        # Get list of active agents.
        active_robots = list()
        for index in self.distribution.keys():
            if len(self.distribution[index].path[0]) > 0:
                active_robots.append(index)

        # Best joit policy.
        best_joint_policy = dict()

        # Loop throughh each agent sequentially and choose the action sequence that maximises the intermediate payoff.
        action_sequences = dict()
        for index in active_robots:
            best_payoff = -1
            for i in range(len(self.distribution[index].path)):
                action_sequences[index] = self.distribution[index].path[i].copy()
                intermediate_payoff = self.f_reward(action_sequences)
                if intermediate_payoff > best_payoff:
                    best_payoff = intermediate_payoff
                    best_joint_policy[index] = self.distribution[index].path[i].copy()
            action_sequences[index] = best_joint_policy[index].copy()
        self.best_response = deepcopy(best_joint_policy)

