import numpy as np
import os
import sys
import platform
import pytest
import gym

def test_deep_q_neural_network(device_id):
    if False:
        print('Hello World!')
    if platform.system() != 'Linux':
        pytest.skip('test only runs on Linux (Gym Atari dependency)')
    abs_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(abs_path)
    sys.path.append(os.path.join(abs_path, '..', '..', '..', '..', 'Examples', 'ReinforcementLearning'))
    dqn = __import__('DeepQNeuralNetwork')
    ENV_NAME = 'Pong-v3'
    env = gym.make(ENV_NAME)
    agent = dqn.DeepQAgent((4, 84, 84), env.action_space.n, train_after=100, memory_size=1000, monitor=False)
    current_step = 0
    max_steps = 1000
    current_state = dqn.as_ale_input(env.reset())
    while current_step < max_steps:
        action = agent.act(current_state)
        (new_state, reward, done, _) = env.step(action)
        new_state = dqn.as_ale_input(new_state)
        reward = np.clip(reward, -1, 1)
        agent.observe(current_state, action, reward, done)
        agent.train()
        current_state = new_state
        if done:
            current_state = dqn.as_ale_input(env.reset())
        current_step += 1
    assert len(agent._memory) == 1000