
def get_action_space_size(brain):
    return brain.vector_action_space_size


def get_state_space_size(env):
    states = env.vector_observations
    return states.shape[1]


def get_information_about_env(env_info, brain):
    # number of agents
    num_agents = len(env_info.agents)
    # size of each action
    action_size = get_action_space_size(brain)
    # examine the state space
    state_size = get_state_space_size(env_info)
    return num_agents, action_size, state_size
