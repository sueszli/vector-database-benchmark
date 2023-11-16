import copy
from typing import Optional
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.typing import EnvConfigDict

@PublicAPI
class EnvContext(dict):
    """Wraps env configurations to include extra rllib metadata.

    These attributes can be used to parameterize environments per process.
    For example, one might use `worker_index` to control which data file an
    environment reads in on initialization.

    RLlib auto-sets these attributes when constructing registered envs.
    """

    def __init__(self, env_config: EnvConfigDict, worker_index: int, vector_index: int=0, remote: bool=False, num_workers: Optional[int]=None, recreated_worker: bool=False):
        if False:
            while True:
                i = 10
        'Initializes an EnvContext instance.\n\n        Args:\n            env_config: The env\'s configuration defined under the\n                "env_config" key in the Algorithm\'s config.\n            worker_index: When there are multiple workers created, this\n                uniquely identifies the worker the env is created in.\n                0 for local worker, >0 for remote workers.\n            vector_index: When there are multiple envs per worker, this\n                uniquely identifies the env index within the worker.\n                Starts from 0.\n            remote: Whether individual sub-environments (in a vectorized\n                env) should be @ray.remote actors or not.\n            num_workers: The total number of (remote) workers in the set.\n                0 if only a local worker exists.\n            recreated_worker: Whether the worker that holds this env is a recreated one.\n                This means that it replaced a previous (failed) worker when\n                `recreate_failed_workers=True` in the Algorithm\'s config.\n        '
        dict.__init__(self, env_config)
        self.worker_index = worker_index
        self.vector_index = vector_index
        self.remote = remote
        self.num_workers = num_workers
        self.recreated_worker = recreated_worker

    def copy_with_overrides(self, env_config: Optional[EnvConfigDict]=None, worker_index: Optional[int]=None, vector_index: Optional[int]=None, remote: Optional[bool]=None, num_workers: Optional[int]=None, recreated_worker: Optional[bool]=None) -> 'EnvContext':
        if False:
            print('Hello World!')
        "Returns a copy of this EnvContext with some attributes overridden.\n\n        Args:\n            env_config: Optional env config to use. None for not overriding\n                the one from the source (self).\n            worker_index: Optional worker index to use. None for not\n                overriding the one from the source (self).\n            vector_index: Optional vector index to use. None for not\n                overriding the one from the source (self).\n            remote: Optional remote setting to use. None for not overriding\n                the one from the source (self).\n            num_workers: Optional num_workers to use. None for not overriding\n                the one from the source (self).\n            recreated_worker: Optional flag, indicating, whether the worker that holds\n                the env is a recreated one. This means that it replaced a previous\n                (failed) worker when `recreate_failed_workers=True` in the Algorithm's\n                config.\n\n        Returns:\n            A new EnvContext object as a copy of self plus the provided\n            overrides.\n        "
        return EnvContext(copy.deepcopy(env_config) if env_config is not None else self, worker_index if worker_index is not None else self.worker_index, vector_index if vector_index is not None else self.vector_index, remote if remote is not None else self.remote, num_workers if num_workers is not None else self.num_workers, recreated_worker if recreated_worker is not None else self.recreated_worker)

    def set_defaults(self, defaults: dict) -> None:
        if False:
            print('Hello World!')
        'Sets missing keys of self to the values given in `defaults`.\n\n        If `defaults` contains keys that already exist in self, don\'t override\n        the values with these defaults.\n\n        Args:\n            defaults: The key/value pairs to add to self, but only for those\n                keys in `defaults` that don\'t exist yet in self.\n\n        .. testcode::\n            :skipif: True\n\n            from ray.rllib.env.env_context import EnvContext\n            env_ctx = EnvContext({"a": 1, "b": 2}, worker_index=0)\n            env_ctx.set_defaults({"a": -42, "c": 3})\n            print(env_ctx)\n\n        .. testoutput::\n\n            {"a": 1, "b": 2, "c": 3}\n        '
        for (key, value) in defaults.items():
            if key not in self:
                self[key] = value

    def __str__(self):
        if False:
            return 10
        return super().__str__()[:-1] + f', worker={self.worker_index}/{self.num_workers}, vector_idx={self.vector_index}, remote={self.remote}' + '}'