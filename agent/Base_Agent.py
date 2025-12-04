"""
Base agent implementations for decentralised MCTS agent.
Author: Nhat Nguyen (School of Computer Science - University of Adelaide)

"""


from algos.tree import Tree
from algos.mcts import growTree
from algos.functions import random_choice_bias, packet_error_probability
from envs.prm import Euclindean_dist
from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np
import pandas as pd
import threading
import shared_info
import logging
import time


class TreeExhaustedException(Exception):
    "Raised when the tree can no longer be grown from the current root."
    pass


class OutOfActions(Exception):
    "Raised when no more action can be taken."
    pass


class Base_Agent(threading.Thread, ABC):
    '''Abstract class implementations for decentralised MCTS agent.'''
    def __init__(self, initial_position:list, i:int, n_agents:int, c_p:float, budget:float, planning_time:float, N_components:int, N_com_every:int, logger:logging=None):
        super().__init__()
        # Action budget contrainst.
        self.budget = budget
        # Actions sequences.
        self.actions_sequence = list()
        # Planning time.
        self.planning_time = planning_time
        # Number of exchanged paths.
        self.N_components = N_components
        # Compression cycle.
        self.N_com_every = N_com_every
        # Exploration factor.
        self.c_p = c_p
        # Logger.
        self.logger = logger
        # Best rollout path and score found.
        self.best_score = 0
        self.best_rollout = list()
        # Starting position in the graph.
        self.position = initial_position
        # Number of agents and the robot's index.
        self.index = i
        self.n_agents = n_agents
        self.active_agents = np.array(range(0, n_agents), dtype=int)
        # Index of the current root node.
        self.root_idx = 0
        # Path histories and probabilities.
        self.distribution = dict()
        for agent in self.active_agents:
            self.distribution[agent] = pd.DataFrame(data={'prob': [1], 'path': [list()], 'tree_idx': [0], 'iter': [-np.inf]})
        # Best respone of the joint policies.
        self.best_response = dict()
        # Communication stats.
        self.com_ratio = 0
        self.com_times = 0
        # Iteration counter.
        self.iter = 0
        # Planning data.
        self.rollout_history = list()

    def set_root_idx(self, new_root_idx:int):
        '''Update the root node.'''
        self.root_idx = new_root_idx

    def set_active_agents(self, new_active_agents):
        '''Update list of active agents.'''
        self.active_agents = new_active_agents

    def set_position(self, new_position):
        '''Update curent position.'''
        self.position = new_position

    def get_current_state(self):
        '''Get the state of action.'''
        return self.tree.data.state[self.root_idx]

    def set_tree(self, state, action):
        '''Each agent holds a MCTS tree.'''
        self.tree = Tree(state=state,
                        actions_to_try=[action],
                        score=0.0,
                        N=0.0,
                        last_backdrop_epoch=np.nan,
                        best_rollout_score=-np.inf,
                        best_rollout_path=[list()])

    def reset_tree(self, state, action):
        '''Restart the search tree.'''
        self.root_idx = 0
        self.iter = 0
        self.set_tree(state, action)
        self.distribution = dict()
        for agent in self.active_agents:
            self.distribution[agent] = pd.DataFrame(data={'prob': [1], 'path': [list()], 'tree_idx': [0], 'iter': [-np.inf]})

    def get_actions_sequence(self):
        '''Return the actions sequence with highest probability in the probability distribution table.'''
        max_idx = self.distribution[self.index].prob.idxmax()
        return self.distribution[self.index].path[max_idx]

    def get_state_action(self):
        '''Get the immediate action and its index.'''
        # Index of path with highest prob.
        best_path_idx = self.distribution[self.index].prob.idxmax()
        # Tree index of best path.
        max_idx = self.distribution[self.index].tree_idx[best_path_idx]
        # No more action can be taken.
        if max_idx == self.root_idx:
            raise OutOfActions
        # The new root is the index of the first node, the action is the node state.
        new_root_idx = self.tree.get_seq_indice(max_idx, self.root_idx)[1]
        action = self.tree.data.state[new_root_idx]
        return action, new_root_idx

    def compress_tree(self, number_of_components:int):
        '''Compress the search tree.'''
        # Get the current depth of the root node.
        current_depth = int(self.tree.data.depth[self.root_idx])
        # If current depth is 0, get the whole tree.
        if current_depth == 0:
            sub_tree = self.tree.data[1:]
        # The subtree must matches every predecessor nodes.
        else:
            sub_tree = self.tree.data.loc[self.tree.data.apply(lambda row: (row["depth"] > current_depth) and (row["best_rollout_path"][1:current_depth+1] == self.actions_sequence[:current_depth]), axis=1)]

        # Number of nodes to be sent.
        immediate_children = sub_tree.copy()
        n = min(number_of_components, immediate_children.shape[0])
        if n > 0:
            # Sorted the tree.
            sent_children = self.sort_node(immediate_children, n)
            # Tree indices to be sent.
            tree_idx = sent_children.index
            # Redistributed the probabilites based on their scores.
            initial_probability = np.finfo(float).eps + sent_children.score
            initial_probability /= sum(initial_probability)
            # Each compressed tree contains n paths and their probabilities.
            self.distribution[self.index] = pd.DataFrame(data={'prob': initial_probability, 
                                                    'path': self.tree.data.best_rollout_path[tree_idx], 
                                                    'tree_idx': tree_idx,
                                                    'iter': [self.iter for _ in range(n)]}).reset_index(drop=True)
        else:
            if self.iter != 0:
                # The tree is exhausted within the given budget.
                self.distribution[self.index] = pd.DataFrame(data={'prob': [1], 
                                                        'path': [self.actions_sequence], 
                                                        'tree_idx': [self.root_idx],
                                                        'iter': [self.iter]}).reset_index(drop=True)
                raise TreeExhaustedException

    def simulate_other_robots(self):
        '''Simulate other robot paths based on their probabilities.'''
        self.best_response.clear()
        for i in self.active_agents:
            if i != self.index:
                self.best_response[i] = (random_choice_bias(
                                                self.distribution[i]['path'].copy().tolist(),
                                                self.distribution[i]['prob'].copy().tolist()))

    def updateDistribution(self, alpha, beta):
        '''Update the probabilities of choosing action sequences.'''
        idx = self.distribution[self.index]['tree_idx']
        reward = np.array(deepcopy(self.tree.data['score'][idx]))
        prob = np.array(deepcopy(self.distribution[self.index]['prob']))
        # Replace approaching-zero values.
        prob = np.where(prob < 1e-10, 1e-10, prob)

        overal_expectation = np.sum(np.multiply(prob, reward))
        log_probability = np.log(prob)
        entropy = np.sum(np.multiply(prob, log_probability))

        log_gradient = -alpha * ((overal_expectation - reward)/beta + entropy + log_probability)
        # Clipping to avoid exp overflown.
        log_sum = np.clip(log_probability + log_gradient, -709, 709)

        self.distribution[self.index].prob = np.exp(log_sum)
        self.distribution[self.index].prob /= sum(self.distribution[self.index].prob)
        self.distribution[self.index].iter = self.iter

    def communicate(self):
        '''Communicate the distribution with others.'''
        # Check what agents are in comm range.
        comm_idx = list()
        with shared_info.global_pos_lock:
            for other_idx in self.active_agents:
                if other_idx != self.index:
                    distance_to_agent = Euclindean_dist(self.position, shared_info.global_pos[other_idx])
                    if np.random.rand() > packet_error_probability(distance_to_agent):
                        comm_idx.append(other_idx)
        # for other_idx in self.active_agents:
        #     if other_idx != self.index:
        #         comm_idx.append(other_idx)
        # Communicate to other robots.
        with shared_info.global_distribution_lock:
            shared_info.global_distribution[self.index] = deepcopy(self.distribution)
            for idx in comm_idx:
                for other_idx in shared_info.global_distribution[idx].keys():
                    if other_idx != self.index and other_idx in self.active_agents:
                        # Discard if other's info is older.
                        if shared_info.global_distribution[idx][other_idx].iter[0] > self.distribution[other_idx].iter[0]:
                            self.distribution[other_idx] = deepcopy(shared_info.global_distribution[idx][other_idx])
        # Update communication stats.
        if len(self.active_agents) > 1:
            self.com_ratio = (self.com_ratio * self.com_times + len(comm_idx)/(len(self.active_agents)-1))/(self.com_times + 1)
            self.com_times += 1

    def execute_action_update_state(self):
        '''Find the immediate action and tree index with highest probabilites. Return True if success.'''
        # if self.logger:
        #     self.logger.info("Agent {} executing...".format(self.index))
        try:
            action, new_root_idx  = self.get_state_action()
        # Terminate if no more action can be taken.
        except OutOfActions:
            # if self.logger:
            #     self.logger.info("Agent {} terminating early...".format(self.index))
            return False
        self.actions_sequence.append(action)
        # Set the new root.
        self.set_root_idx(new_root_idx)
        # Set new position.
        self.update_position()
        # Synchronise share data
        with shared_info.global_pos_lock:
            shared_info.global_pos[self.index] = self.position
        with shared_info.joint_path_lock:
            shared_info.joint_path[self.index].append(action)
        return True

    @abstractmethod
    def update_position(self):
        '''Method to update the current position.'''
        pass

    @abstractmethod
    def sort_node(self):
        '''Method to sort the nodes to be sent.'''
        pass

    @abstractmethod
    def update_belief(self):
        '''Method to update belief about the environment.'''
        pass

    @abstractmethod
    def f_backprop(self):
        '''Backpropabation method.'''
        pass

    @abstractmethod
    def f_score(self):
        '''Utility function.'''
        pass

    @abstractmethod
    def f_actions(self):
        '''Available functions to take.'''
        pass

    @abstractmethod
    def f_terminal(self):
        '''Terminal condition.'''
        pass

    @abstractmethod
    def f_sampler(self):
        '''Random rollout policy.'''
        pass

    def f_ucb(self, Np:float, Nc:float):
        '''UCB policy.'''
        return 2 * self.c_p * np.sqrt(np.divide(np.log(Np), Nc))

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
                    # Simulate other robots actions.
                    self.simulate_other_robots()
                    # Re-initialise backprop.
                    f_backprop = lambda tree, tree_idx, current_score, current_rollout: self.f_backprop(tree, tree_idx, current_score, current_rollout)
                    # Grow the tree search.
                    _, rollout_path = growTree(self.tree, self.root_idx, self.f_score, self.f_actions, self.f_terminal, self.f_ucb, self.f_sampler, f_backprop)
                    self.rollout_history.append(rollout_path)
                # Update the distribution and share with others.
                self.updateDistribution(alpha=1, beta=max(np.power(0.95, self.iter), 0.001))
                self.communicate()

            # Compress the tree into product distribution.
            try:
                self.compress_tree(self.N_components)
            except TreeExhaustedException:
                self.communicate()
                break

    def run(self):
        '''Run the simulation until terminal condition met.'''
        while not self.f_terminal(self.actions_sequence):
            # Refine belief of the rewards distribution.
            self.update_belief()
            # Start planning.
            self.planning()
            # Take one action, terminate early if no more action can be taken.
            if not self.execute_action_update_state():
                break
        if self.logger:
            # self.logger.info("Agent {} finished".format(self.index))
            self.logger.info("Agent {} com ratio {}".format(self.index, self.com_ratio))

    def __str__(self):
        return str(self.tree) + "," + str(self.distribution) + "," + str(self.best_score) + "," + str(self.best_rollout)

