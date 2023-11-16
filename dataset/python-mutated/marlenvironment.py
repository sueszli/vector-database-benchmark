""" Example MARL Environment for RLLIB SUMO Utlis

    Author: Lara CODECA lara.codeca@gmail.com

    See:
        https://github.com/lcodeca/rllibsumoutils
        https://github.com/lcodeca/rllibsumodocker
    for further details.
"""
import collections
import logging
import os
import sys
from pprint import pformat
from numpy.random import RandomState
import gymnasium as gym
from ray.rllib.env import MultiAgentEnv
from ray.rllib.examples.simulators.sumo.utils import SUMOUtils, sumo_default_config
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
    import traci.constants as tc
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
logger = logging.getLogger(__name__)

def env_creator(config):
    if False:
        while True:
            i = 10
    'Environment creator used in the environment registration.'
    logger.info('Environment creation: SUMOTestMultiAgentEnv')
    return SUMOTestMultiAgentEnv(config)
MS_TO_KMH = 3.6

class SUMOSimulationWrapper(SUMOUtils):
    """A wrapper for the interaction with the SUMO simulation"""

    def _initialize_simulation(self):
        if False:
            for i in range(10):
                print('nop')
        'Specific simulation initialization.'
        try:
            super()._initialize_simulation()
        except NotImplementedError:
            pass

    def _initialize_metrics(self):
        if False:
            return 10
        'Specific metrics initialization'
        try:
            super()._initialize_metrics()
        except NotImplementedError:
            pass
        self.veh_subscriptions = dict()
        self.collisions = collections.defaultdict(int)

    def _default_step_action(self, agents):
        if False:
            while True:
                i = 10
        'Specific code to be executed in every simulation step'
        try:
            super()._default_step_action(agents)
        except NotImplementedError:
            pass
        collisions = self.traci_handler.simulation.getCollidingVehiclesIDList()
        logger.debug('Collisions: %s', pformat(collisions))
        for veh in collisions:
            self.collisions[veh] += 1
        self.veh_subscriptions = self.traci_handler.vehicle.getAllSubscriptionResults()
        for (veh, vals) in self.veh_subscriptions.items():
            logger.debug('Subs: %s, %s', pformat(veh), pformat(vals))
        running = set()
        for agent in agents:
            if agent in self.veh_subscriptions:
                running.add(agent)
        if len(running) == 0:
            logger.info('All the agent left the simulation..')
            self.end_simulation()
        return True

class SUMOAgent:
    """Agent implementation."""

    def __init__(self, agent, config):
        if False:
            for i in range(10):
                print('nop')
        self.agent_id = agent
        self.config = config
        self.action_to_meaning = dict()
        for (pos, action) in enumerate(config['actions']):
            self.action_to_meaning[pos] = config['actions'][action]
        logger.debug("Agent '%s' configuration \n %s", self.agent_id, pformat(self.config))

    def step(self, action, sumo_handler):
        if False:
            print('Hello World!')
        'Implements the logic of each specific action passed as input.'
        logger.debug('Agent %s: action %d', self.agent_id, action)
        logger.debug('Subscriptions: %s', pformat(sumo_handler.veh_subscriptions[self.agent_id]))
        previous_speed = sumo_handler.veh_subscriptions[self.agent_id][tc.VAR_SPEED]
        new_speed = previous_speed + self.action_to_meaning[action]
        logger.debug('Before %.2f', previous_speed)
        sumo_handler.traci_handler.vehicle.setSpeed(self.agent_id, new_speed)
        logger.debug('After %.2f', new_speed)
        return

    def reset(self, sumo_handler):
        if False:
            return 10
        'Resets the agent and return the observation.'
        route = '{}_rou'.format(self.agent_id)
        sumo_handler.traci_handler.route.add(route, ['road'])
        sumo_handler.traci_handler.vehicle.add(self.agent_id, route, departLane='best', departSpeed='max')
        sumo_handler.traci_handler.vehicle.subscribeLeader(self.agent_id)
        sumo_handler.traci_handler.vehicle.subscribe(self.agent_id, varIDs=[tc.VAR_SPEED])
        logger.info('Agent %s reset done.', self.agent_id)
        return (self.agent_id, self.config['start'])
DEFAULT_SCENARIO_CONFING = {'sumo_config': sumo_default_config(), 'agent_rnd_order': True, 'log_level': 'WARN', 'seed': 42, 'misc': {'max_distance': 5000}}
DEFAULT_AGENT_CONFING = {'origin': 'road', 'destination': 'road', 'start': 0, 'actions': {'acc': 1.0, 'none': 0.0, 'dec': -1.0}, 'max_speed': 130}

class SUMOTestMultiAgentEnv(MultiAgentEnv):
    """
    A RLLIB environment for testing MARL environments with SUMO simulations.
    """

    def __init__(self, config):
        if False:
            return 10
        'Initialize the environment.'
        super(SUMOTestMultiAgentEnv, self).__init__()
        self._config = config
        level = logging.getLevelName(config['scenario_config']['log_level'])
        logger.setLevel(level)
        self.simulation = None
        self.rndgen = RandomState(config['scenario_config']['seed'])
        self.agents_init_list = dict()
        self.agents = dict()
        for (agent, agent_config) in self._config['agent_init'].items():
            self.agents[agent] = SUMOAgent(agent, agent_config)
        self.resetted = True
        self.episodes = 0
        self.steps = 0

    def seed(self, seed):
        if False:
            while True:
                i = 10
        'Set the seed of a possible random number generator.'
        self.rndgen = RandomState(seed)

    def get_agents(self):
        if False:
            print('Hello World!')
        'Returns a list of the agents.'
        return self.agents.keys()

    def __del__(self):
        if False:
            return 10
        logger.info('Environment destruction: SUMOTestMultiAgentEnv')
        if self.simulation:
            del self.simulation

    def get_observation(self, agent):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the observation of a given agent.\n        See http://sumo.sourceforge.net/pydoc/traci._simulation.html\n        '
        speed = 0
        distance = self._config['scenario_config']['misc']['max_distance']
        if agent in self.simulation.veh_subscriptions:
            speed = round(self.simulation.veh_subscriptions[agent][tc.VAR_SPEED] * MS_TO_KMH)
            leader = self.simulation.veh_subscriptions[agent][tc.VAR_LEADER]
            if leader:
                (veh, dist) = leader
                if veh:
                    distance = round(dist)
        ret = [speed, distance]
        logger.debug('Agent %s --> Obs: %s', agent, pformat(ret))
        return ret

    def compute_observations(self, agents):
        if False:
            while True:
                i = 10
        'For each agent in the list, return the observation.'
        obs = dict()
        for agent in agents:
            obs[agent] = self.get_observation(agent)
        return obs

    def get_reward(self, agent):
        if False:
            i = 10
            return i + 15
        'Return the reward for a given agent.'
        speed = self.agents[agent].config['max_speed']
        if agent in self.simulation.veh_subscriptions:
            speed = round(self.simulation.veh_subscriptions[agent][tc.VAR_SPEED] * MS_TO_KMH)
        logger.debug('Agent %s --> Reward %d', agent, speed)
        return speed

    def compute_rewards(self, agents):
        if False:
            i = 10
            return i + 15
        'For each agent in the list, return the rewards.'
        rew = dict()
        for agent in agents:
            rew[agent] = self.get_reward(agent)
        return rew

    def reset(self, *, seed=None, options=None):
        if False:
            i = 10
            return i + 15
        'Resets the env and returns observations from ready agents.'
        self.resetted = True
        self.episodes += 1
        self.steps = 0
        if self.simulation:
            del self.simulation
        self.simulation = SUMOSimulationWrapper(self._config['scenario_config']['sumo_config'])
        waiting_agents = list()
        for agent in self.agents.values():
            (agent_id, start) = agent.reset(self.simulation)
            waiting_agents.append((start, agent_id))
        waiting_agents.sort()
        starting_time = waiting_agents[0][0]
        self.simulation.fast_forward(starting_time)
        self.simulation._default_step_action(self.agents.keys())
        initial_obs = self.compute_observations(self.agents.keys())
        return (initial_obs, {})

    def step(self, action_dict):
        if False:
            i = 10
            return i + 15
        '\n        Returns observations from ready agents.\n\n        The returns are dicts mapping from agent_id strings to values. The\n        number of agents in the env can vary over time.\n\n        Returns\n        -------\n            obs: New observations for each ready agent.\n            rewards: Reward values for each ready agent. If the\n                episode is just started, the value will be None.\n            dones: Done values for each ready agent. The special key\n                "__all__" (required) is used to indicate env termination.\n            infos: Optional info values for each agent id.\n        '
        self.resetted = False
        self.steps += 1
        logger.debug('====> [SUMOTestMultiAgentEnv:step] Episode: %d - Step: %d <====', self.episodes, self.steps)
        dones = {}
        dones['__all__'] = False
        shuffled_agents = sorted(action_dict.keys())
        if self._config['scenario_config']['agent_rnd_order']:
            logger.debug('Shuffling the order of the agents.')
            self.rndgen.shuffle(shuffled_agents)
        for agent in shuffled_agents:
            self.agents[agent].step(action_dict[agent], self.simulation)
        logger.debug('Before SUMO')
        ongoing_simulation = self.simulation.step(until_end=False, agents=set(action_dict.keys()))
        logger.debug('After SUMO')
        if not ongoing_simulation:
            logger.info('Reached the end of the SUMO simulation.')
            dones['__all__'] = True
        (obs, rewards, infos) = ({}, {}, {})
        for agent in action_dict:
            if self.simulation.collisions[agent] > 0:
                dones[agent] = True
                obs[agent] = [0, 0]
                rewards[agent] = -self.agents[agent].config['max_speed']
                self.simulation.traci_handler.remove(agent, reason=tc.REMOVE_VAPORIZED)
            else:
                dones[agent] = agent not in self.simulation.veh_subscriptions
                obs[agent] = self.get_observation(agent)
                rewards[agent] = self.get_reward(agent)
        logger.debug('Observations: %s', pformat(obs))
        logger.debug('Rewards: %s', pformat(rewards))
        logger.debug('Dones: %s', pformat(dones))
        logger.debug('Info: %s', pformat(infos))
        logger.debug('========================================================')
        return (obs, rewards, dones, dones, infos)

    def get_action_space_size(self, agent):
        if False:
            for i in range(10):
                print('nop')
        'Returns the size of the action space.'
        return len(self.agents[agent].config['actions'])

    def get_action_space(self, agent):
        if False:
            while True:
                i = 10
        'Returns the action space.'
        return gym.spaces.Discrete(self.get_action_space_size(agent))

    def get_set_of_actions(self, agent):
        if False:
            for i in range(10):
                print('nop')
        'Returns the set of possible actions for an agent.'
        return set(range(self.get_action_space_size(agent)))

    def get_obs_space_size(self, agent):
        if False:
            print('Hello World!')
        'Returns the size of the observation space.'
        return (self.agents[agent].config['max_speed'] + 1) * (self._config['scenario_config']['misc']['max_distance'] + 1)

    def get_obs_space(self, agent):
        if False:
            return 10
        'Returns the observation space.'
        return gym.spaces.MultiDiscrete([self.agents[agent].config['max_speed'] + 1, self._config['scenario_config']['misc']['max_distance'] + 1])