"""Reinforcement Learning (RL) Agent Base for Open Spiel."""
import abc
import collections
StepOutput = collections.namedtuple('step_output', ['action', 'probs'])

class AbstractAgent(metaclass=abc.ABCMeta):
    """Abstract base class for Open Spiel RL agents."""

    @abc.abstractmethod
    def __init__(self, player_id, session=None, observation_spec=None, name='agent', **agent_specific_kwargs):
        if False:
            while True:
                i = 10
        'Initializes agent.\n\n    Args:\n      player_id: integer, mandatory. Corresponds to the player position in the\n        game and is used to index the observation list.\n      session: optional Tensorflow session.\n      observation_spec: optional dict containing observation specifications.\n      name: string. Must be used to scope TF variables. Defaults to `agent`.\n      **agent_specific_kwargs: optional extra args.\n    '

    @abc.abstractmethod
    def step(self, time_step, is_evaluation=False):
        if False:
            for i in range(10):
                print('nop')
        'Returns action probabilities and chosen action at `time_step`.\n\n    Agents should handle `time_step` and extract the required part of the\n    `time_step.observations` field. This flexibility enables algorithms which\n    rely on opponent observations / information, e.g. CFR.\n\n    `is_evaluation` can be used so agents change their behaviour for evaluation\n    purposes, e.g.: preventing exploration rate decaying during test and\n    insertion of data to replay buffers.\n\n    Arguments:\n      time_step: an instance of rl_environment.TimeStep.\n      is_evaluation: bool indicating whether the step is an evaluation routine,\n        as opposed to a normal training step.\n\n    Returns:\n      A `StepOutput` for the current `time_step`.\n    '