"""
Centralised MCTS planner for multi-agent.
Author: Nhat Nguyen (School of Computer Science - University of Adelaide)

"""


from algos.tree import Tree
from algos.mcts import growTree
from envs.graph import Graph
from algos.functions import random_choice_bias, evaluate_immediate_actions
from abc import ABC, abstractmethod
import numpy as np
import time


class Central_Agent(ABC):
    '''Centralised MCTS planner for multi-agent.'''
    def __init__(self, initial_state:float, actions_to_try:dict, n_agents:int, c_p:float, budget:float, planning_time:float):
        # Action budget contrainst.
        self.budget = budget
        # Planning time.
        self.planning_time = planning_time
        # Exploration factor.
        self.c_p = c_p
        # Number of agents.
        self.n_agents = n_agents
        # Actions to try for each agent.
        self.actions_to_try = actions_to_try
        # Index of the current root node.
        self.root_idx = 0
        # Single tree for every agents.
        self.tree = Tree(state=initial_state,
                        actions_to_try=[self.actions_to_try[0]],
                        score=0.0,
                        N=0.0,
                        best_rollout_score=-np.inf,
                        best_rollout_path=[list()])

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
    
    def f_joint(self, edge_history:list):
        '''Joint path histories of all robots.'''
        edge_history_robots = dict()
        for i in range(self.n_agents):
            edge_history_robots[i] = list()
        for i in range(1, len(edge_history)):
            edge_history_robots[(i - 1) % self.n_agents].append(edge_history[i])
        return edge_history_robots

    def planning(self):
        '''Planning method.'''
        # Planning start.
        start = time.time()
        while (time.time() - start) < self.planning_time:
            # Grow the tree search.
            rollout_score, rollout_history = growTree(self.tree, self.root_idx, self.f_score, self.f_actions, self.f_terminal, self.f_ucb, self.f_sampler)

            if rollout_score > self.tree.data.at[0, 'best_rollout_score']:
                self.tree.data.at[0, 'best_rollout_score'] = rollout_score
                self.tree.data.at[0, 'best_rollout_path'] = self.f_joint(rollout_history)


class Central_InfoGathering_Agent(Central_Agent):
    '''Central MCTS planner for information gathering task.'''
    def __init__(self, initial_state: float, actions_to_try: list, n_agents: int, c_p: float, budget: float, planning_time: float, Z: Graph, n_rewards:int):
        super().__init__(initial_state, actions_to_try, n_agents, c_p, budget, planning_time)
        self.Z = Z
        self.n_rewards = n_rewards

    def f_reward(self, edge_history):
        '''Global objective function.'''
        # return sum(self.Z.evaluate_traj_reward(edge_history))/self.n_rewards
        return self.Z.evaluate_traj_reward(edge_history)/self.n_rewards

    def f_score(self, edge_history):
        '''Global objective function.'''
        # return sum(self.Z.evaluate_traj_reward(self.f_joint(edge_history)))/self.n_rewards
        return self.Z.evaluate_traj_reward(self.f_joint(edge_history))/self.n_rewards

    def f_actions(self, edge_history):
        '''Available functions are listed of successor nodes.'''
        return self.actions_to_try[len(edge_history)-1] if len(edge_history) <= self.n_agents else self.Z.find_edge(edge_history[-self.n_agents])

    def f_terminal(self, edge_history):
        '''Terminal condition is when the travelling budget of the agent expires.'''
        edge_history_robots = self.f_joint(edge_history)
        idx = (len(edge_history)+1) % self.n_agents
        robot_history = edge_history_robots[idx]
        return True if self.Z.evaluate_traj_cost(robot_history) > self.budget else False

    def f_sampler(self, available_actions):
        '''Pick next action based on its immediate reward (greedy random).'''
        return random_choice_bias(available_actions, evaluate_immediate_actions(self.Z, available_actions))

