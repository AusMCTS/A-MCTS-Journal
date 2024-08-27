"""
Dec-MCTS agent implementations.
Author: Nhat Nguyen (School of Computer Science - University of Adelaide)

"""


from algos.tree import Tree
from agent.Base_Agent import Base_Agent
import numpy as np
import logging


class DecMCTS_Agent(Base_Agent):
    '''Abstract Dec-MCTS agent implementations.'''
    def __init__(self, initial_state:float, initial_actions:list, initial_position:list, i:int, n_agents:int, gamma:float, c_p:float,budget:float, planning_time:float, N_components:int, N_com_every:int, logger:logging=None):
        super().__init__(initial_position, i, n_agents, c_p, budget, planning_time, N_components, N_com_every, logger)
        # Each robot holds a MCTS.
        self.set_tree(initial_state, initial_actions)
        # Forgetting factor.
        self.forgetting_factor = gamma

    def sort_node(self, immediate_children, n):
        '''Sorted the tree based on empirical score in descending order.'''
        return immediate_children.nlargest(n, 'score')

    def f_backprop(self, tree:Tree, tree_idx:int, current_score:float, current_path:list):
        '''Backpropagation for decentralised MCTS.'''
        # Node sequences to be backpropagated.
        backtrace = tree.get_seq_indice(tree_idx, self.root_idx)

        # Retrieve all data for calculations.
        old_score = np.array(tree.data.score[backtrace].copy())
        last_backprop = np.array(tree.data.last_backdrop_epoch[backtrace].copy())
        current_n = np.array(tree.data.N[backtrace].copy())
        # If the node has just been added.
        np.nan_to_num(last_backprop, copy=False, nan=self.iter)

        # Update new score and number of time node being selected using the dec-mcst formula.
        discount_factor = np.power(self.forgetting_factor, (self.iter - last_backprop))
        discount_n = np.multiply(discount_factor, current_n)
        new_accumulative_score = np.multiply(discount_n, old_score) + current_score
        new_n = discount_n + 1
        tree.data.loc[backtrace, 'score'] = np.divide(new_accumulative_score, new_n)
        tree.data.loc[backtrace, 'N'] = new_n

        # Update best score, best path and current epoch.
        to_replace = list()
        for i in backtrace:
            if current_score > tree.data.loc[i, 'best_rollout_score']:
                to_replace.append(i)
                tree.data.at[i, 'best_rollout_path'] = current_path
        tree.data.loc[to_replace, 'best_rollout_score'] = current_score
        tree.data.loc[backtrace, 'last_backdrop_epoch'] = self.iter

