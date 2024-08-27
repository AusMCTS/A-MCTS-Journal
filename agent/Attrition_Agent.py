"""
Attrition agent implementations.
Author: Nhat Nguyen (School of Computer Science - University of Adelaide)

"""


from agent.Base_Agent import Base_Agent
from agent.InfoGathering_Agent import A_InfoGathering_Agent, Dec_InfoGathering_Agent, SW_InfoGathering_Agent, Greedy_InfoGathering_Agent
from envs.graph import Graph
from algos.functions import f_joint
from abc import abstractmethod
import numpy as np
import pandas as pd
import logging
import shared_info


class Attrition_Agent(Base_Agent):
    '''Base class for agent with attrition risk.'''
    def run(self):
        '''Run the simulation until terminal condition met or agent is killed.'''
        while not self.f_terminal(self.actions_sequence):
            # Refine belief of the rewards distribution.
            self.update_belief()
            # Start planning.
            self.planning()
            # Take one action, terminate early if no more action can be taken.
            if not self.execute_action_update_state():
                break
            # Check if agent is stopped.
            if self.stop():
                # Clear info in global distribution.
                with shared_info.global_distribution_lock:
                    for i in range(self.n_agents):
                        shared_info.global_distribution[self.index][i] = pd.DataFrame(data={'prob': [1], 'path': [list()], 'tree_idx': [0], 'iter': [-np.inf]})
                if self.logger:
                    self.logger.info("Agent {} is killed.".format(self.index))
                break
        if self.logger:
            # self.logger.info("Agent {} finished".format(self.index))
            self.logger.info("Agent {} com ratio {}".format(self.index, self.com_ratio))

    @abstractmethod
    def stop(self):
        '''Method to stop the agent.'''
        pass


class InfoGathering_Attrtion_Agent(Attrition_Agent):
    '''Base class for information gathering task with attrition risk.'''
    def set_fail_condition(self, fail_conidtion: any = None):
        # Condition for agent to fail.
        self.fail_condition = fail_conidtion

    def stop(self):
        '''Method to stop the agent. In this implementation we use action sequence length.'''
        if self.fail_condition:
            if self.Z.evaluate_traj_cost(self.actions_sequence) >= self.fail_condition:
                return True
        return False


class A_AttritionGathering_Agent(A_InfoGathering_Agent, InfoGathering_Attrtion_Agent):
    '''A-MCTS agent for information gathering task with attrition risk.'''
    def __init__(self, initial_state: float, initial_actions: list, initial_position: list, i: int, n_agents: int, gamma: float, c_p: float, budget: float, planning_time: float, N_components: int, N_com_every: int, Z: Graph, n_rewards: int, RM_iter: int, msg_threshold: int = 0, logger: logging = None, fail_conidtion: any = None):
        super().__init__(initial_state, initial_actions, initial_position, i, n_agents, gamma, c_p, budget, planning_time, N_components, N_com_every, Z, n_rewards, RM_iter, msg_threshold, logger)
        InfoGathering_Attrtion_Agent.set_fail_condition(self, fail_conidtion=fail_conidtion)


class Greedy_AttritionGathering_Agent(Greedy_InfoGathering_Agent, InfoGathering_Attrtion_Agent):
    '''Greedy-MCTS agent for information gathering task with attrition risk.'''
    def __init__(self, initial_state: float, initial_actions: list, initial_position: list, i: int, n_agents: int, gamma: float, c_p: float, budget: float, planning_time: float, N_components: int, N_com_every: int, Z: Graph, n_rewards: int, RM_iter: int, msg_threshold: int = 0, logger: logging = None, fail_conidtion: any = None):
        super().__init__(initial_state, initial_actions, initial_position, i, n_agents, gamma, c_p, budget, planning_time, N_components, N_com_every, Z, n_rewards, RM_iter, msg_threshold, logger)
        InfoGathering_Attrtion_Agent.set_fail_condition(self, fail_conidtion=fail_conidtion)


class Dec_AttritionGathering_Agent(Dec_InfoGathering_Agent, InfoGathering_Attrtion_Agent):
    '''Dec-MCTS agent for information gathering task with attrition risk.'''
    def __init__(self, initial_state: float, initial_actions: list, initial_position: list, i: int, n_agents: int, gamma: float, c_p: float, budget: float, planning_time: float, N_components: int, N_com_every: int, Z: Graph, n_rewards: int, logger: logging = None, fail_conidtion: any = None):
        super().__init__(initial_state, initial_actions, initial_position, i, n_agents, gamma, c_p, budget, planning_time, N_components, N_com_every, Z, n_rewards, logger)
        InfoGathering_Attrtion_Agent.set_fail_condition(self, fail_conidtion=fail_conidtion)


class SW_AttritioGathering_Agent(SW_InfoGathering_Agent, InfoGathering_Attrtion_Agent):
    '''SW-MCTS agent for information gathering task with attrition risk.'''
    def __init__(self, initial_state: float, initial_actions: list, initial_position: list, i: int, n_agents: int, max_window: int, c_p: float, budget: float, planning_time: float, N_components: int, N_com_every: int, Z: Graph, n_rewards: int, logger: logging = None, fail_conidtion: any = None):
        super().__init__(initial_state, initial_actions, initial_position, i, n_agents, max_window, c_p, budget, planning_time, N_components, N_com_every, Z, n_rewards, logger)
        InfoGathering_Attrtion_Agent.set_fail_condition(self, fail_conidtion=fail_conidtion)


class Global_AttritionGathering_Agent(Dec_AttritionGathering_Agent):
    '''Dec-MCTS agent for information gathering task with attrition risk using global utility.'''
    def __init__(self, initial_state: float, initial_actions: list, initial_position: list, i: int, n_agents: int, gamma: float, c_p: float, budget: float, planning_time: float, N_components: int, N_com_every: int, Z: Graph, n_rewards: int, logger: logging = None, fail_conidtion: any = None):
        super().__init__(initial_state, initial_actions, initial_position, i, n_agents, gamma, c_p, budget, planning_time, N_components, N_com_every, Z, n_rewards, logger, fail_conidtion)

    def f_score(self, edge_history):
        ''' Utility is the global utility.'''
        return self.f_reward(f_joint(self.index, edge_history, self.best_response))


class Reset_AttritionGathering_Agent(Dec_AttritionGathering_Agent):
    '''Dec-MCTS with reset agent for information gathering task with attrition risk.'''
    def __init__(self, initial_state: float, initial_actions: list, initial_position: list, i: int, n_agents: int, gamma: float, c_p: float, budget: float, planning_time: float, N_components: int, N_com_every: int, Z: Graph, n_rewards: int, logger: logging = None, fail_conidtion: any = None):
        super().__init__(initial_state, initial_actions, initial_position, i, n_agents, gamma, c_p, budget, planning_time, N_components, N_com_every, Z, n_rewards, logger, fail_conidtion)
        with shared_info.status_lock:
            shared_info.status[self.index] = True

    def check_status(self):
        '''Remove agent info if it has not communicated in a while.'''
        flag = False
        idx_to_remove = list()
        with shared_info.status_lock:
            for i in self.active_agents:
                if i != self.index and not shared_info.status[i]:
                    idx_to_remove.append(i)
                    flag = True
        if flag:
            # Clear info in global distribution.
            with shared_info.global_distribution_lock:
                for i in idx_to_remove:
                    shared_info.global_distribution[self.index][i] = pd.DataFrame(data={'prob': [1], 'path': [list()], 'tree_idx': [0], 'iter': [-np.inf]})
            # Remove inactive agents.
            self.active_agents = np.setdiff1d(self.active_agents, idx_to_remove)
            # Remaining budget.
            self.budget -= self.Z.evaluate_traj_cost(self.actions_sequence)
            # Remaining budget until stop.
            if self.fail_condition:
                self.fail_condition -= self.Z.evaluate_traj_cost(self.actions_sequence)
            # Reset tree.
            self.reset_tree(self.actions_sequence[-1], self.f_actions(self.actions_sequence))
            self.actions_sequence = list()
        return flag

    def run(self):
        '''Run the simulation until terminal condition met or agent is killed.'''
        while not self.f_terminal(self.actions_sequence):
            # Refine belief of the rewards distribution.
            self.update_belief()
            # Start planning.
            self.planning()
            # Take one action, terminate early if no more action can be taken.
            if not self.execute_action_update_state():
                break
            # Check if agent is stopped.
            if self.stop():
                # Clear info in global distribution.
                with shared_info.global_distribution_lock:
                    for i in self.active_agents:
                        shared_info.global_distribution[self.index][i] = pd.DataFrame(data={'prob': [1], 'path': [list()], 'tree_idx': [0], 'iter': [-np.inf]})
                # Change active status.
                with shared_info.status_lock:
                    shared_info.status[self.index] = False
                if self.logger:
                    self.logger.info("Agent {} is killed.".format(self.index))
                break
            # Check if reset planning.
            else:
                if self.check_status():
                    if self.logger:
                        self.logger.info("Agent {} reset.".format(self.index))
        if self.logger:
            # self.logger.info("Agent {} finished".format(self.index))
            self.logger.info("Agent {} com ratio {}".format(self.index, self.com_ratio))

