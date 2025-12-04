"""
Helper functions for MCTS.
Author: Nhat Nguyen (School of Computer Science - University of Adelaide)

Reference:
Dec-MCTS: Decentralized planning for multi-robot active perception
[https://doi.org/10.1177/0278364918755924]
"""


from envs.graph import Graph
import numpy as np
import math


def random_choice(vector:list):
    '''Random choice among all available options.'''
    idx = np.random.randint(low=0, high=len(vector), size=1)[0]
    choice = vector[idx]
    return choice


def random_choice_bias(choices:list, pb:list):
    '''Random choice with probabilities.'''
    unnorm_pb = pb
    cuml_pb = np.cumsum(unnorm_pb)
    random_draw = cuml_pb[-1] * np.random.rand()
    idx = np.where(random_draw < cuml_pb)[0][0]
    return choices[idx]


def f_joint(rob_index:int, edge_history:list, edge_history_other_robots:dict):
    '''Joint path histories of all robots.'''
    edge_history_other_robots.update({rob_index: edge_history})
    return edge_history_other_robots


def evaluate_immediate_actions(G:Graph, available_actions:list, packet_size:int=1):
    '''Get reward mask of the edge between the current node to all immediate successor nodes.'''
    pb = list()
    for i in range(len(available_actions)):
        pb.append(1 + sum(G.evaluate_edge_reward(available_actions[i]) >= packet_size))
    return pb


def exhaustive_search(robots, active_robots, f_payoff):
    """Exhaustive search through the whole search space M^N."""
    # Get the number of agents and the number of action sequences.
    num_agents = len(active_robots)
    num_actions = 0
    for index in active_robots:
        if len(robots[index].distribution[index].prob) > num_actions:
            num_actions = len(robots[index].distribution[index].prob)
    action_sequences = dict()

    num_iterations = pow(num_actions, num_agents)

    # Best joit policy and global payoff.
    best_joint_policy = dict()
    best_payoff = 0
    j = 0

    # Compute every possible combinations.
    for i in range(num_iterations):
        action_sequences.clear()
        for index in active_robots:
            action_idx = math.floor(i / pow(num_actions, num_agents - j - 1) % num_actions)
            if action_idx < len(robots[index].distribution[index].prob):
                action_sequences[index] = robots[index].distribution[index].path[action_idx].copy()
            j = (j + 1) % num_agents

        payoff = f_payoff(action_sequences)
        # Return the best combination.
        if payoff > best_payoff:
            best_payoff = payoff
            best_joint_policy = action_sequences.copy()

    return best_joint_policy


def absorption_efficiency(frequency:float=13):
    '''Absorption efficiency.
    @param frequency - kHz'''
    squared_frequency = np.power(frequency, 2)
    absorption_efficiency = 0.11 * np.divide(squared_frequency, 1 + squared_frequency) + 44 * np.divide(squared_frequency, 4100 + squared_frequency) + 2.75 * (10**-4) * squared_frequency + 0.003
    return np.power(10, absorption_efficiency/10)


def path_loss(distance:float, frequency:float=13):
    '''Path loss.
    @param distance - km
    @param frequency - kHz'''
    k = 1.5 # propagation loss
    path_loss = np.power(distance, k) * np.power(absorption_efficiency(frequency), distance)
    return path_loss


def noise_level(w:float, frequency:float=13):
    '''Noise power.
    @param w - wind speed
    @param frequency - kHz'''
    s = 0.5   # surface shipping activity [0,1]
    N_t = 17 - 30 * np.log10(frequency)
    N_s = 40 + 20 * (s - 0.5) + 26 * np.log10(frequency) - 60 * np.log10(frequency + 0.03)
    N_w = 50 + 7.5 * np.power(w, 0.5) + 20 * np.log10(frequency) - 40 * np.log10(frequency + 0.4)
    N_th = -15 + 20 * np.log10(frequency)
    return np.power(10, N_t/10) + np.power(10, N_s/10) + np.power(10, N_w/10) + np.power(10, N_th/10)


def signal_noise_ratio(w:float, distance:float, frequency:float=13):
    '''Signal to noise ratio.
    @param distance - km
    @param frequency - kHz'''
    P = 1   # transmissted power (W)
    B = 1   # bandwidth power (kHz)
    SNR = P / (noise_level(w, frequency) * path_loss(distance, frequency) * B * 10**-10.6)
    return SNR


def packet_error_probability(distance:float, w:float=2.5, frequency:float=13):
    '''Packet error rate.
    @param distance - km
    @param frequency - kHz'''
    packet_size = 256
    # If distance is very small, just return 0 error rate.
    if np.isclose(distance, 0):
        return 0
    # Create a temp variable to prevent runtime overflow.
    temp = max(0, 1 - np.divide(1, 4*signal_noise_ratio(w, distance, frequency)))
    return 1 - np.power(temp, packet_size)

