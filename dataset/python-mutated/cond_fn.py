"""Defines many boolean functions indicating when to step and reset.
"""
import tensorflow as tf
import gin.tf

@gin.configurable
def env_transition(agent, state, action, transition_type, environment_steps, num_episodes):
    if False:
        return 10
    'True if the transition_type is TRANSITION or FINAL_TRANSITION.\n\n  Args:\n    agent: RL agent.\n    state: A [num_state_dims] tensor representing a state.\n    action: Action performed.\n    transition_type: Type of transition after action\n    environment_steps: Number of steps performed by environment.\n    num_episodes: Number of episodes.\n  Returns:\n    cond: Returns an op that evaluates to true if the transition type is\n    not RESTARTING\n  '
    del agent, state, action, num_episodes, environment_steps
    cond = tf.logical_not(transition_type)
    return cond

@gin.configurable
def env_restart(agent, state, action, transition_type, environment_steps, num_episodes):
    if False:
        print('Hello World!')
    'True if the transition_type is RESTARTING.\n\n  Args:\n    agent: RL agent.\n    state: A [num_state_dims] tensor representing a state.\n    action: Action performed.\n    transition_type: Type of transition after action\n    environment_steps: Number of steps performed by environment.\n    num_episodes: Number of episodes.\n  Returns:\n    cond: Returns an op that evaluates to true if the transition type equals\n    RESTARTING.\n  '
    del agent, state, action, num_episodes, environment_steps
    cond = tf.identity(transition_type)
    return cond

@gin.configurable
def every_n_steps(agent, state, action, transition_type, environment_steps, num_episodes, n=150):
    if False:
        return 10
    'True once every n steps.\n\n  Args:\n    agent: RL agent.\n    state: A [num_state_dims] tensor representing a state.\n    action: Action performed.\n    transition_type: Type of transition after action\n    environment_steps: Number of steps performed by environment.\n    num_episodes: Number of episodes.\n    n: Return true once every n steps.\n  Returns:\n    cond: Returns an op that evaluates to true if environment_steps\n    equals 0 mod n. We increment the step before checking this condition, so\n    we do not need to add one to environment_steps.\n  '
    del agent, state, action, transition_type, num_episodes
    cond = tf.equal(tf.mod(environment_steps, n), 0)
    return cond

@gin.configurable
def every_n_episodes(agent, state, action, transition_type, environment_steps, num_episodes, n=2, steps_per_episode=None):
    if False:
        print('Hello World!')
    'True once every n episodes.\n\n  Specifically, evaluates to True on the 0th step of every nth episode.\n  Unlike environment_steps, num_episodes starts at 0, so we do want to add\n  one to ensure it does not reset on the first call.\n\n  Args:\n    agent: RL agent.\n    state: A [num_state_dims] tensor representing a state.\n    action: Action performed.\n    transition_type: Type of transition after action\n    environment_steps: Number of steps performed by environment.\n    num_episodes: Number of episodes.\n    n: Return true once every n episodes.\n    steps_per_episode: How many steps per episode. Needed to determine when a\n    new episode starts.\n  Returns:\n    cond: Returns an op that evaluates to true on the last step of the episode\n      (i.e. if num_episodes equals 0 mod n).\n  '
    assert steps_per_episode is not None
    del agent, action, transition_type
    ant_fell = tf.logical_or(state[2] < 0.2, state[2] > 1.0)
    cond = tf.logical_and(tf.logical_or(ant_fell, tf.equal(tf.mod(num_episodes + 1, n), 0)), tf.equal(tf.mod(environment_steps, steps_per_episode), 0))
    return cond

@gin.configurable
def failed_reset_after_n_episodes(agent, state, action, transition_type, environment_steps, num_episodes, steps_per_episode=None, reset_state=None, max_dist=1.0, epsilon=1e-10):
    if False:
        return 10
    'Every n episodes, returns True if the reset agent fails to return.\n\n  Specifically, evaluates to True if the distance between the state and the\n  reset state is greater than max_dist at the end of the episode.\n\n  Args:\n    agent: RL agent.\n    state: A [num_state_dims] tensor representing a state.\n    action: Action performed.\n    transition_type: Type of transition after action\n    environment_steps: Number of steps performed by environment.\n    num_episodes: Number of episodes.\n    steps_per_episode: How many steps per episode. Needed to determine when a\n    new episode starts.\n    reset_state: State to which the reset controller should return.\n    max_dist: Agent is considered to have successfully reset if its distance\n    from the reset_state is less than max_dist.\n    epsilon: small offset to ensure non-negative/zero distance.\n  Returns:\n    cond: Returns an op that evaluates to true if num_episodes+1 equals 0\n    mod n. We add one to the num_episodes so the environment is not reset after\n    the 0th step.\n  '
    assert steps_per_episode is not None
    assert reset_state is not None
    del agent, state, action, transition_type, num_episodes
    dist = tf.sqrt(tf.reduce_sum(tf.squared_difference(state, reset_state)) + epsilon)
    cond = tf.logical_and(tf.greater(dist, tf.constant(max_dist)), tf.equal(tf.mod(environment_steps, steps_per_episode), 0))
    return cond

@gin.configurable
def q_too_small(agent, state, action, transition_type, environment_steps, num_episodes, q_min=0.5):
    if False:
        for i in range(10):
            print('nop')
    'True of q is too small.\n\n  Args:\n    agent: RL agent.\n    state: A [num_state_dims] tensor representing a state.\n    action: Action performed.\n    transition_type: Type of transition after action\n    environment_steps: Number of steps performed by environment.\n    num_episodes: Number of episodes.\n    q_min: Returns true if the qval is less than q_min\n  Returns:\n    cond: Returns an op that evaluates to true if qval is less than q_min.\n  '
    del transition_type, environment_steps, num_episodes
    state_for_reset_agent = tf.stack(state[:-1], tf.constant([0], dtype=tf.float))
    qval = agent.BASE_AGENT_CLASS.critic_net(tf.expand_dims(state_for_reset_agent, 0), tf.expand_dims(action, 0))[0, :]
    cond = tf.greater(tf.constant(q_min), qval)
    return cond

@gin.configurable
def true_fn(agent, state, action, transition_type, environment_steps, num_episodes):
    if False:
        for i in range(10):
            print('nop')
    'Returns an op that evaluates to true.\n\n  Args:\n    agent: RL agent.\n    state: A [num_state_dims] tensor representing a state.\n    action: Action performed.\n    transition_type: Type of transition after action\n    environment_steps: Number of steps performed by environment.\n    num_episodes: Number of episodes.\n  Returns:\n    cond: op that always evaluates to True.\n  '
    del agent, state, action, transition_type, environment_steps, num_episodes
    cond = tf.constant(True, dtype=tf.bool)
    return cond

@gin.configurable
def false_fn(agent, state, action, transition_type, environment_steps, num_episodes):
    if False:
        return 10
    'Returns an op that evaluates to false.\n\n  Args:\n    agent: RL agent.\n    state: A [num_state_dims] tensor representing a state.\n    action: Action performed.\n    transition_type: Type of transition after action\n    environment_steps: Number of steps performed by environment.\n    num_episodes: Number of episodes.\n  Returns:\n    cond: op that always evaluates to False.\n  '
    del agent, state, action, transition_type, environment_steps, num_episodes
    cond = tf.constant(False, dtype=tf.bool)
    return cond