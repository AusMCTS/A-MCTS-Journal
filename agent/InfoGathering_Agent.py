"""
Information gathering agent implementations.
Author: Nhat Nguyen (School of Computer Science - University of Adelaide)

"""


from agent.Base_Agent import Base_Agent
from agent.SwMCTS_Agent import SwMCTS_Agent
from agent.DecMCTS_Agent import DecMCTS_Agent
from agent.AMCTS_Agent import AMCTS_Agent, GreedyMCTS_Agent
from algos.functions import random_choice_bias, evaluate_immediate_actions, f_joint
from envs.graph import Graph
import logging


class InfoGathering_Agent(Base_Agent):
    '''Base class for information gathering applications.'''
    def set_belief(self, Z:Graph, n_rewards:int):
        # Map of the environment.
        self.Z = Z
        self.n_rewards = n_rewards

    def update_position(self):
        '''Update the current position on the graph.'''
        latest_action = self.actions_sequence[-1]
        current_node = self.Z.edges_list[int(latest_action)][1]
        self.set_position(self.Z.find_coordinates(current_node))

    def f_reward(self, edge_history):
        '''Global objective function.'''
        return sum(self.Z.evaluate_traj_reward(edge_history))/self.n_rewards

    def f_score(self, edge_history):
        '''Utility is the joint reward minus other robots' reward.'''
        other_robots_reward = self.f_reward(self.best_response)
        return self.f_reward(f_joint(self.index, edge_history, self.best_response)) - other_robots_reward

    def f_actions(self, edge_history):
        '''Available functions are listed of successor nodes.'''
        return self.Z.find_edge(edge_history[-1])

    def f_terminal(self, edge_history):
        '''Terminal condition is when the travelling budget expires.'''
        return self.Z.evaluate_traj_cost(edge_history) > self.budget

    def f_sampler(self, available_actions):
        '''Pick next action based on its immediate reward (greedy random).'''
        return random_choice_bias(available_actions, evaluate_immediate_actions(self.Z, available_actions))

    def update_belief(self):
        '''Method to update belief about the environment.'''
        pass


class SW_InfoGathering_Agent(SwMCTS_Agent, InfoGathering_Agent):
    '''Sw-MCTS agent for information gathering task.'''
    def __init__(self, initial_state: float, initial_actions: list, initial_position: list, i: int, n_agents: int, max_window: int, c_p: float, budget: float, planning_time: float, N_components: int, N_com_every: int, Z: Graph, n_rewards:int, logger: logging = None):
        super().__init__(initial_state, initial_actions, initial_position, i, n_agents, max_window, c_p, budget, planning_time, N_components, N_com_every, logger)
        InfoGathering_Agent.set_belief(self, Z=Z, n_rewards=n_rewards)


class Dec_InfoGathering_Agent(DecMCTS_Agent, InfoGathering_Agent):
    '''Dec-MCTS agent for information gathering task.'''
    def __init__(self, initial_state: float, initial_actions: list, initial_position: list, i: int, n_agents: int, gamma: float, c_p: float, budget: float, planning_time: float, N_components: int, N_com_every: int, Z: Graph, n_rewards:int, logger: logging = None):
        super().__init__(initial_state, initial_actions, initial_position, i, n_agents, gamma, c_p, budget, planning_time, N_components, N_com_every, logger)
        InfoGathering_Agent.set_belief(self, Z=Z, n_rewards=n_rewards)


class A_InfoGathering_Agent(AMCTS_Agent, InfoGathering_Agent):
    '''A-MCTS agent for information gathering task.'''
    def __init__(self, initial_state: float, initial_actions: list, initial_position: list, i: int, n_agents: int, gamma: float, c_p: float, budget: float, planning_time: float, N_components: int, N_com_every: int, Z: Graph, n_rewards:int, RM_iter: int, msg_threshold: int = 0, logger: logging = None):
        super().__init__(initial_state, initial_actions, initial_position, i, n_agents, gamma, c_p, budget, planning_time, N_components, N_com_every, RM_iter, msg_threshold, logger)
        InfoGathering_Agent.set_belief(self, Z=Z, n_rewards=n_rewards)

    def f_score(self, edge_history):
        ''' Utility is the global utility.'''
        return self.f_reward(f_joint(self.index, edge_history, self.best_response))


class Greedy_InfoGathering_Agent(GreedyMCTS_Agent, InfoGathering_Agent):
    '''Greedy-MCTS agent for information gathering task.'''
    def __init__(self, initial_state: float, initial_actions: list, initial_position: list, i: int, n_agents: int, gamma: float, c_p: float, budget: float, planning_time: float, N_components: int, N_com_every: int, Z: Graph, n_rewards:int,  RM_iter: int, msg_threshold: int = 0, logger: logging = None):
        super().__init__(initial_state, initial_actions, initial_position, i, n_agents, gamma, c_p, budget, planning_time, N_components, N_com_every, RM_iter, msg_threshold, logger)
        InfoGathering_Agent.set_belief(self, Z=Z, n_rewards=n_rewards)

    def f_score(self, edge_history):
        ''' Utility is the global utility.'''
        return self.f_reward(f_joint(self.index, edge_history, self.best_response))

