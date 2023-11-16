"""Models for RunConfig and its related components."""
from types import SimpleNamespace

class RunConfig(SimpleNamespace):
    """Class for Run Configuration.

    Attributes:
        shots (int): the number of shots
        seed_simulator (int): the seed to use in the simulator
        memory (bool): whether to request memory from backend (per-shot
            readouts)
        parameter_binds (list[dict]): List of parameter bindings
    """

    def __init__(self, shots=None, seed_simulator=None, memory=None, parameter_binds=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Initialize a RunConfig object\n\n        Args:\n            shots (int): the number of shots\n            seed_simulator (int): the seed to use in the simulator\n            memory (bool): whether to request memory from backend\n                (per-shot readouts)\n            parameter_binds (list[dict]): List of parameter bindings\n            **kwargs: optional fields\n        '
        if shots is not None:
            self.shots = shots
        if seed_simulator is not None:
            self.seed_simulator = seed_simulator
        if memory is not None:
            self.memory = memory
        if parameter_binds is not None:
            self.parameter_binds = parameter_binds
        self.__dict__.update(kwargs)

    @classmethod
    def from_dict(cls, data):
        if False:
            print('Hello World!')
        'Create a new RunConfig object from a dictionary.\n\n        Args:\n            data (dict): A dictionary representing the RunConfig to create.\n                         It will be in the same format as output by\n                         :meth:`to_dict`.\n\n        Returns:\n            RunConfig: The RunConfig from the input dictionary.\n        '
        return cls(**data)

    def to_dict(self):
        if False:
            while True:
                i = 10
        'Return a dictionary format representation of the RunConfig\n\n        Returns:\n            dict: The dictionary form of the RunConfig.\n        '
        return self.__dict__