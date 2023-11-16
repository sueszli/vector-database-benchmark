from collections import defaultdict, deque
import logging
import platform
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type
import ray
from ray.actor import ActorClass, ActorHandle
logger = logging.getLogger(__name__)

class TaskPool:
    """Helper class for tracking the status of many in-flight actor tasks."""

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._tasks = {}
        self._objects = {}
        self._fetching = deque()

    def add(self, worker, all_obj_refs):
        if False:
            return 10
        if isinstance(all_obj_refs, list):
            obj_ref = all_obj_refs[0]
        else:
            obj_ref = all_obj_refs
        self._tasks[obj_ref] = worker
        self._objects[obj_ref] = all_obj_refs

    def completed(self, blocking_wait=False):
        if False:
            return 10
        pending = list(self._tasks)
        if pending:
            (ready, _) = ray.wait(pending, num_returns=len(pending), timeout=0)
            if not ready and blocking_wait:
                (ready, _) = ray.wait(pending, num_returns=1, timeout=10.0)
            for obj_ref in ready:
                yield (self._tasks.pop(obj_ref), self._objects.pop(obj_ref))

    def completed_prefetch(self, blocking_wait=False, max_yield=999):
        if False:
            for i in range(10):
                print('nop')
        'Similar to completed but only returns once the object is local.\n\n        Assumes obj_ref only is one id.'
        for (worker, obj_ref) in self.completed(blocking_wait=blocking_wait):
            self._fetching.append((worker, obj_ref))
        for _ in range(max_yield):
            if not self._fetching:
                break
            yield self._fetching.popleft()

    def reset_workers(self, workers):
        if False:
            return 10
        'Notify that some workers may be removed.'
        for (obj_ref, ev) in self._tasks.copy().items():
            if ev not in workers:
                del self._tasks[obj_ref]
                del self._objects[obj_ref]
        for _ in range(len(self._fetching)):
            (ev, obj_ref) = self._fetching.popleft()
            if ev in workers:
                self._fetching.append((ev, obj_ref))

    @property
    def count(self):
        if False:
            while True:
                i = 10
        return len(self._tasks)

def create_colocated_actors(actor_specs: Sequence[Tuple[Type, Any, Any, int]], node: Optional[str]='localhost', max_attempts: int=10) -> Dict[Type, List[ActorHandle]]:
    if False:
        while True:
            i = 10
    'Create co-located actors of any type(s) on any node.\n\n    Args:\n        actor_specs: Tuple/list with tuples consisting of: 1) The\n            (already @ray.remote) class(es) to construct, 2) c\'tor args,\n            3) c\'tor kwargs, and 4) the number of actors of that class with\n            given args/kwargs to construct.\n        node: The node to co-locate the actors on. By default ("localhost"),\n            place the actors on the node the caller of this function is\n            located on. Use None for indicating that any (resource fulfilling)\n            node in the cluster may be used.\n        max_attempts: The maximum number of co-location attempts to\n            perform before throwing an error.\n\n    Returns:\n        A dict mapping the created types to the list of n ActorHandles\n        created (and co-located) for that type.\n    '
    if node == 'localhost':
        node = platform.node()
    ok = [[] for _ in range(len(actor_specs))]
    for attempt in range(max_attempts):
        all_good = True
        for (i, (typ, args, kwargs, count)) in enumerate(actor_specs):
            args = args or []
            kwargs = kwargs or {}
            if len(ok[i]) < count:
                co_located = try_create_colocated(cls=typ, args=args, kwargs=kwargs, count=count * (attempt + 1), node=node)
                if node is None:
                    node = ray.get(co_located[0].get_host.remote())
                ok[i].extend(co_located)
                if len(ok[i]) < count:
                    all_good = False
            if len(ok[i]) > count:
                for a in ok[i][count:]:
                    a.__ray_terminate__.remote()
                ok[i] = ok[i][:count]
        if all_good:
            return ok
    raise Exception('Unable to create enough colocated actors -> aborting.')

def try_create_colocated(cls: Type[ActorClass], args: List[Any], count: int, kwargs: Optional[List[Any]]=None, node: Optional[str]='localhost') -> List[ActorHandle]:
    if False:
        for i in range(10):
            print('nop')
    'Tries to co-locate (same node) a set of Actors of the same type.\n\n    Returns a list of successfully co-located actors. All actors that could\n    not be co-located (with the others on the given node) will not be in this\n    list.\n\n    Creates each actor via it\'s remote() constructor and then checks, whether\n    it has been co-located (on the same node) with the other (already created)\n    ones. If not, terminates the just created actor.\n\n    Args:\n        cls: The Actor class to use (already @ray.remote "converted").\n        args: List of args to pass to the Actor\'s constructor. One item\n            per to-be-created actor (`count`).\n        count: Number of actors of the given `cls` to construct.\n        kwargs: Optional list of kwargs to pass to the Actor\'s constructor.\n            One item per to-be-created actor (`count`).\n        node: The node to co-locate the actors on. By default ("localhost"),\n            place the actors on the node the caller of this function is\n            located on. If None, will try to co-locate all actors on\n            any available node.\n\n    Returns:\n        List containing all successfully co-located actor handles.\n    '
    if node == 'localhost':
        node = platform.node()
    kwargs = kwargs or {}
    actors = [cls.remote(*args, **kwargs) for _ in range(count)]
    (co_located, non_co_located) = split_colocated(actors, node=node)
    logger.info('Got {} colocated actors of {}'.format(len(co_located), count))
    for a in non_co_located:
        a.__ray_terminate__.remote()
    return co_located

def split_colocated(actors: List[ActorHandle], node: Optional[str]='localhost') -> Tuple[List[ActorHandle], List[ActorHandle]]:
    if False:
        return 10
    'Splits up given actors into colocated (on same node) and non colocated.\n\n    The co-location criterion depends on the `node` given:\n    If given (or default: platform.node()): Consider all actors that are on\n    that node "colocated".\n    If None: Consider the largest sub-set of actors that are all located on\n    the same node (whatever that node is) as "colocated".\n\n    Args:\n        actors: The list of actor handles to split into "colocated" and\n            "non colocated".\n        node: The node defining "colocation" criterion. If provided, consider\n            thos actors "colocated" that sit on this node. If None, use the\n            largest subset within `actors` that are sitting on the same\n            (any) node.\n\n    Returns:\n        Tuple of two lists: 1) Co-located ActorHandles, 2) non co-located\n        ActorHandles.\n    '
    if node == 'localhost':
        node = platform.node()
    hosts = ray.get([a.get_host.remote() for a in actors])
    if node is None:
        node_groups = defaultdict(set)
        for (host, actor) in zip(hosts, actors):
            node_groups[host].add(actor)
        max_ = -1
        largest_group = None
        for host in node_groups:
            if max_ < len(node_groups[host]):
                max_ = len(node_groups[host])
                largest_group = host
        non_co_located = []
        for host in node_groups:
            if host != largest_group:
                non_co_located.extend(list(node_groups[host]))
        return (list(node_groups[largest_group]), non_co_located)
    else:
        co_located = []
        non_co_located = []
        for (host, a) in zip(hosts, actors):
            if host == node:
                co_located.append(a)
            else:
                non_co_located.append(a)
        return (co_located, non_co_located)

def drop_colocated(actors: List[ActorHandle]) -> List[ActorHandle]:
    if False:
        return 10
    (colocated, non_colocated) = split_colocated(actors)
    for a in colocated:
        a.__ray_terminate__.remote()
    return non_colocated