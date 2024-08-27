"""
SW-MCTS agent implementations.
Author: Nhat Nguyen (School of Computer Science - University of Adelaide)

"""


from collections import deque
from algos.tree import Tree
from agent.Base_Agent import Base_Agent
import numpy as np
import logging


class SwMCTS_Agent(Base_Agent):
    '''Abstract SW-MCTS agent implementations.'''
    def __init__(self, initial_state:float, initial_actions:list, initial_position:list, i:int, n_agents:int, max_window:int, c_p:float,budget:float, planning_time:float, N_components:int, N_com_every:int, logger:logging=None):
        super().__init__(initial_position, i, n_agents, c_p, budget, planning_time, N_components, N_com_every, logger)
        # Each robot holds a MCTS.
        self.set_tree(initial_state, initial_actions)
        # Forgetting factor.
        self.forgetting_factor = max_window
        # Queue of previous score.
        self.set_window()

    def set_window(self):
        '''Initialise the sliding window.'''
        self.previous_score = deque(maxlen=self.forgetting_factor)

    def set_window_size(self, new_window):
        '''Adjust the maximal window size.'''
        self.forgetting_factor = new_window

    def sort_node(self, immediate_children, n):
        '''Sorted the tree based on number of times node are chosen and score in descending order.'''
        return immediate_children.nlargest(n, ['N', 'score'])

    def f_backprop(self, tree:Tree, tree_idx:int, current_score:int, current_path:list):
        '''Backpropagation for SW-MCTS.'''
        # Node sequences to be backpropagated.
        backtrace = tree.get_seq_indice(tree_idx, self.root_idx)

        # Update the sequence of actions and scores queue.
        # previous score: [[score at t, [list of nodes selected at t]], ...]
        self.previous_score.append((current_score, backtrace))
        for node in backtrace:
            sw_n = 0
            sw_accumulative_score = 0
            for i in range(max(0, len(self.previous_score) - self.forgetting_factor), len(self.previous_score)):
                if node in self.previous_score[i][1]:
                    sw_n += 1
                    sw_accumulative_score += self.previous_score[i][0]
            tree.data.loc[node, 'score'] = np.divide(sw_accumulative_score, sw_n)
            tree.data.loc[node, 'N'] = sw_n

        # Update best score, best path and current epoch.
        to_replace = list()
        for i in backtrace:
            if (abs(self.iter - tree.data.loc[i, 'last_backdrop_epoch']) > self.forgetting_factor) or (current_score > tree.data.loc[i, 'best_rollout_score']):
                to_replace.append(i)
                tree.data.at[i, 'best_rollout_path'] = current_path
        tree.data.loc[to_replace, 'best_rollout_score'] = current_score
        tree.data.loc[backtrace, 'last_backdrop_epoch'] = self.iter

