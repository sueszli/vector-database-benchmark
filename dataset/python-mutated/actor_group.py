import weakref
from dataclasses import dataclass
import logging
from typing import List, TypeVar, Optional, Dict, Type, Tuple
import ray
from ray.actor import ActorHandle
from ray.util.annotations import Deprecated
from ray._private.utils import get_ray_doc_version
T = TypeVar('T')
ActorMetadata = TypeVar('ActorMetadata')
logger = logging.getLogger(__name__)

@dataclass
class ActorWrapper:
    """Class containing an actor and its metadata."""
    actor: ActorHandle
    metadata: ActorMetadata

@dataclass
class ActorConfig:
    num_cpus: float
    num_gpus: float
    resources: Optional[Dict[str, float]]
    init_args: Tuple
    init_kwargs: Dict

class ActorGroupMethod:

    def __init__(self, actor_group: 'ActorGroup', method_name: str):
        if False:
            return 10
        self.actor_group = weakref.ref(actor_group)
        self._method_name = method_name

    def __call__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        raise TypeError(f"ActorGroup methods cannot be called directly. Instead of running 'object.{self._method_name}()', try 'object.{self._method_name}.remote()'.")

    def remote(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        return [getattr(a.actor, self._method_name).remote(*args, **kwargs) for a in self.actor_group().actors]

@Deprecated(message=f'For stateless/task processing, use ray.util.multiprocessing, see details in https://docs.ray.io/en/{get_ray_doc_version()}/ray-more-libs/multiprocessing.html. For stateful/actor processing such as batch prediction, use Datasets.map_batches(compute=ActorPoolStrategy, ...), see details in https://docs.ray.io/en/{get_ray_doc_version()}/data/api/dataset.html#ray.data.Dataset.map_batches.', warning=True)
class ActorGroup:
    """Group of Ray Actors that can execute arbitrary functions.

    ``ActorGroup`` launches Ray actors according to the given
    specification. It can then execute arbitrary Python functions in each of
    these actors.

    If not enough resources are available to launch the actors, the Ray
    cluster will automatically scale up if autoscaling is enabled.

    Args:
        actor_cls: The class to use as the remote actors.
        num_actors: The number of the provided Ray actors to
            launch. Defaults to 1.
        num_cpus_per_actor: The number of CPUs to reserve for each
            actor. Fractional values are allowed. Defaults to 1.
        num_gpus_per_actor: The number of GPUs to reserve for each
            actor. Fractional values are allowed. Defaults to 0.
        resources_per_actor (Optional[Dict[str, float]]):
            Dictionary specifying the resources that will be
            requested for each actor in addition to ``num_cpus_per_actor``
            and ``num_gpus_per_actor``.
        init_args, init_kwargs: If ``actor_cls`` is provided,
            these args will be used for the actor initialization.

    """

    def __init__(self, actor_cls: Type, num_actors: int=1, num_cpus_per_actor: float=1, num_gpus_per_actor: float=0, resources_per_actor: Optional[Dict[str, float]]=None, init_args: Optional[Tuple]=None, init_kwargs: Optional[Dict]=None):
        if False:
            return 10
        from ray._private.usage.usage_lib import record_library_usage
        record_library_usage('util.ActorGroup')
        if num_actors <= 0:
            raise ValueError(f'The provided `num_actors` must be greater than 0. Received num_actors={num_actors} instead.')
        if num_cpus_per_actor < 0 or num_gpus_per_actor < 0:
            raise ValueError(f'The number of CPUs and GPUs per actor must not be negative. Received num_cpus_per_actor={num_cpus_per_actor} and num_gpus_per_actor={num_gpus_per_actor}.')
        self.actors = []
        self.num_actors = num_actors
        self.actor_config = ActorConfig(num_cpus=num_cpus_per_actor, num_gpus=num_gpus_per_actor, resources=resources_per_actor, init_args=init_args or (), init_kwargs=init_kwargs or {})
        self._remote_cls = ray.remote(num_cpus=self.actor_config.num_cpus, num_gpus=self.actor_config.num_gpus, resources=self.actor_config.resources)(actor_cls)
        self.start()

    def __getattr__(self, item):
        if False:
            return 10
        if len(self.actors) == 0:
            raise RuntimeError('This ActorGroup has been shutdown. Please start it again.')
        return ActorGroupMethod(self, item)

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self.actors)

    def __getitem__(self, item):
        if False:
            print('Hello World!')
        return self.actors[item]

    def start(self):
        if False:
            return 10
        'Starts all the actors in this actor group.'
        if self.actors and len(self.actors) > 0:
            raise RuntimeError('The actors have already been started. Please call `shutdown` first if you want to restart them.')
        logger.debug(f'Starting {self.num_actors} actors.')
        self.add_actors(self.num_actors)
        logger.debug(f'{len(self.actors)} actors have successfully started.')

    def shutdown(self, patience_s: float=5):
        if False:
            while True:
                i = 10
        'Shutdown all the actors in this actor group.\n\n        Args:\n            patience_s: Attempt a graceful shutdown\n                of the actors for this many seconds. Fallback to force kill\n                if graceful shutdown is not complete after this time. If\n                this is less than or equal to 0, immediately force kill all\n                actors.\n        '
        logger.debug(f'Shutting down {len(self.actors)} actors.')
        if patience_s <= 0:
            for actor in self.actors:
                ray.kill(actor.actor)
        else:
            done_refs = [w.actor.__ray_terminate__.remote() for w in self.actors]
            (done, not_done) = ray.wait(done_refs, timeout=patience_s)
            if not_done:
                logger.debug('Graceful termination failed. Falling back to force kill.')
                for actor in self.actors:
                    ray.kill(actor.actor)
        logger.debug('Shutdown successful.')
        self.actors = []

    def remove_actors(self, actor_indexes: List[int]):
        if False:
            for i in range(10):
                print('nop')
        'Removes the actors with the specified indexes.\n\n        Args:\n            actor_indexes (List[int]): The indexes of the actors to remove.\n        '
        new_actors = []
        for i in range(len(self.actors)):
            if i not in actor_indexes:
                new_actors.append(self.actors[i])
        self.actors = new_actors

    def add_actors(self, num_actors: int):
        if False:
            print('Hello World!')
        'Adds ``num_actors`` to this ActorGroup.\n\n        Args:\n            num_actors: The number of actors to add.\n        '
        new_actors = []
        new_actor_metadata = []
        for _ in range(num_actors):
            actor = self._remote_cls.remote(*self.actor_config.init_args, **self.actor_config.init_kwargs)
            new_actors.append(actor)
            if hasattr(actor, 'get_actor_metadata'):
                new_actor_metadata.append(actor.get_actor_metadata.remote())
        metadata = ray.get(new_actor_metadata)
        if len(metadata) == 0:
            metadata = [None] * len(new_actors)
        for i in range(len(new_actors)):
            self.actors.append(ActorWrapper(actor=new_actors[i], metadata=metadata[i]))

    @property
    def actor_metadata(self):
        if False:
            print('Hello World!')
        return [a.metadata for a in self.actors]