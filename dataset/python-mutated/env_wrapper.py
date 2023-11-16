from __future__ import annotations
import weakref
from typing import Any, Callable, cast, Dict, Generic, Iterable, Iterator, Optional, Tuple
import gym
from gym import Space
from qlib.rl.aux_info import AuxiliaryInfoCollector
from qlib.rl.interpreter import ActionInterpreter, ObsType, PolicyActType, StateInterpreter
from qlib.rl.reward import Reward
from qlib.rl.simulator import ActType, InitialStateType, Simulator, StateType
from qlib.typehint import TypedDict
from .finite_env import generate_nan_observation
from .log import LogCollector, LogLevel
__all__ = ['InfoDict', 'EnvWrapperStatus', 'EnvWrapper']
SEED_INTERATOR_MISSING = '_missing_'

class InfoDict(TypedDict):
    """The type of dict that is used in the 4th return value of ``env.step()``."""
    aux_info: dict
    'Any information depends on auxiliary info collector.'
    log: Dict[str, Any]
    'Collected by LogCollector.'

class EnvWrapperStatus(TypedDict):
    """
    This is the status data structure used in EnvWrapper.
    The fields here are in the semantics of RL.
    For example, ``obs`` means the observation fed into policy.
    ``action`` means the raw action returned by policy.
    """
    cur_step: int
    done: bool
    initial_state: Optional[Any]
    obs_history: list
    action_history: list
    reward_history: list

class EnvWrapper(gym.Env[ObsType, PolicyActType], Generic[InitialStateType, StateType, ActType, ObsType, PolicyActType]):
    """Qlib-based RL environment, subclassing ``gym.Env``.
    A wrapper of components, including simulator, state-interpreter, action-interpreter, reward.

    This is what the framework of simulator - interpreter - policy looks like in RL training.
    All the components other than policy needs to be assembled into a single object called "environment".
    The "environment" are replicated into multiple workers, and (at least in tianshou's implementation),
    one single policy (agent) plays against a batch of environments.

    Parameters
    ----------
    simulator_fn
        A callable that is the simulator factory.
        When ``seed_iterator`` is present, the factory should take one argument,
        that is the seed (aka initial state).
        Otherwise, it should take zero argument.
    state_interpreter
        State-observation converter.
    action_interpreter
        Policy-simulator action converter.
    seed_iterator
        An iterable of seed. With the help of :class:`qlib.rl.utils.DataQueue`,
        environment workers in different processes can share one ``seed_iterator``.
    reward_fn
        A callable that accepts the StateType and returns a float (at least in single-agent case).
    aux_info_collector
        Collect auxiliary information. Could be useful in MARL.
    logger
        Log collector that collects the logs. The collected logs are sent back to main process,
        via the return value of ``env.step()``.

    Attributes
    ----------
    status : EnvWrapperStatus
        Status indicator. All terms are in *RL language*.
        It can be used if users care about data on the RL side.
        Can be none when no trajectory is available.
    """
    simulator: Simulator[InitialStateType, StateType, ActType]
    seed_iterator: str | Iterator[InitialStateType] | None

    def __init__(self, simulator_fn: Callable[..., Simulator[InitialStateType, StateType, ActType]], state_interpreter: StateInterpreter[StateType, ObsType], action_interpreter: ActionInterpreter[StateType, PolicyActType, ActType], seed_iterator: Optional[Iterable[InitialStateType]], reward_fn: Reward | None=None, aux_info_collector: AuxiliaryInfoCollector[StateType, Any] | None=None, logger: LogCollector | None=None) -> None:
        if False:
            return 10
        for obj in [state_interpreter, action_interpreter, reward_fn, aux_info_collector]:
            if obj is not None:
                obj.env = weakref.proxy(self)
        self.simulator_fn = simulator_fn
        self.state_interpreter = state_interpreter
        self.action_interpreter = action_interpreter
        if seed_iterator is None:
            self.seed_iterator = SEED_INTERATOR_MISSING
        else:
            self.seed_iterator = iter(seed_iterator)
        self.reward_fn = reward_fn
        self.aux_info_collector = aux_info_collector
        self.logger: LogCollector = logger or LogCollector()
        self.status: EnvWrapperStatus = cast(EnvWrapperStatus, None)

    @property
    def action_space(self) -> Space:
        if False:
            for i in range(10):
                print('nop')
        return self.action_interpreter.action_space

    @property
    def observation_space(self) -> Space:
        if False:
            i = 10
            return i + 15
        return self.state_interpreter.observation_space

    def reset(self, **kwargs: Any) -> ObsType:
        if False:
            return 10
        '\n        Try to get a state from state queue, and init the simulator with this state.\n        If the queue is exhausted, generate an invalid (nan) observation.\n        '
        try:
            if self.seed_iterator is None:
                raise RuntimeError('You can trying to get a state from a dead environment wrapper.')
            self.logger.reset()
            if self.seed_iterator is SEED_INTERATOR_MISSING:
                initial_state = None
                self.simulator = cast(Callable[[], Simulator], self.simulator_fn)()
            else:
                initial_state = next(cast(Iterator[InitialStateType], self.seed_iterator))
                self.simulator = self.simulator_fn(initial_state)
            self.status = EnvWrapperStatus(cur_step=0, done=False, initial_state=initial_state, obs_history=[], action_history=[], reward_history=[])
            self.simulator.env = cast(EnvWrapper, weakref.proxy(self))
            sim_state = self.simulator.get_state()
            obs = self.state_interpreter(sim_state)
            self.status['obs_history'].append(obs)
            return obs
        except StopIteration:
            self.seed_iterator = None
            return generate_nan_observation(self.observation_space)

    def step(self, policy_action: PolicyActType, **kwargs: Any) -> Tuple[ObsType, float, bool, InfoDict]:
        if False:
            return 10
        'Environment step.\n\n        See the code along with comments to get a sequence of things happening here.\n        '
        if self.seed_iterator is None:
            raise RuntimeError('State queue is already exhausted, but the environment is still receiving action.')
        self.logger.reset()
        self.status['action_history'].append(policy_action)
        action = self.action_interpreter(self.simulator.get_state(), policy_action)
        self.status['cur_step'] += 1
        self.simulator.step(action)
        done = self.simulator.done()
        self.status['done'] = done
        sim_state = self.simulator.get_state()
        obs = self.state_interpreter(sim_state)
        self.status['obs_history'].append(obs)
        if self.reward_fn is not None:
            rew = self.reward_fn(sim_state)
        else:
            rew = 0.0
        self.status['reward_history'].append(rew)
        if self.aux_info_collector is not None:
            aux_info = self.aux_info_collector(sim_state)
        else:
            aux_info = {}
        if done:
            self.logger.add_scalar('steps_per_episode', self.status['cur_step'])
        self.logger.add_scalar('reward', rew)
        self.logger.add_any('obs', obs, loglevel=LogLevel.DEBUG)
        self.logger.add_any('policy_act', policy_action, loglevel=LogLevel.DEBUG)
        info_dict = InfoDict(log=self.logger.logs(), aux_info=aux_info)
        return (obs, rew, done, info_dict)

    def render(self, mode: str='human') -> None:
        if False:
            return 10
        raise NotImplementedError('Render is not implemented in EnvWrapper.')