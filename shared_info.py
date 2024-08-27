import threading

def init():
    # Share parameters and locks.
    global global_pos, global_pos_lock, global_distribution, global_distribution_lock, joint_path, joint_path_lock, status, status_lock

    # Define global params.
    global_pos = dict()
    global_pos_lock = threading.Lock()
    global_distribution = dict()
    global_distribution_lock = threading.Lock()
    joint_path = dict()
    joint_path_lock = threading.Lock()
    status = dict()
    status_lock = threading.Lock()