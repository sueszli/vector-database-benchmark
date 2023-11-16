from typing import TYPE_CHECKING, Any, Callable, List, TypeVar
import ray
from ray.util.annotations import DeveloperAPI
if TYPE_CHECKING:
    import ray.actor
V = TypeVar('V')

@DeveloperAPI
class ActorPool:
    """Utility class to operate on a fixed pool of actors.

    Arguments:
        actors: List of Ray actor handles to use in this pool.

    Examples:
        .. testcode::

            import ray
            from ray.util.actor_pool import ActorPool

            @ray.remote
            class Actor:
                def double(self, v):
                    return 2 * v

            a1, a2 = Actor.remote(), Actor.remote()
            pool = ActorPool([a1, a2])
            print(list(pool.map(lambda a, v: a.double.remote(v),
                                [1, 2, 3, 4])))

        .. testoutput::

            [2, 4, 6, 8]
    """

    def __init__(self, actors: list):
        if False:
            return 10
        from ray._private.usage.usage_lib import record_library_usage
        record_library_usage('util.ActorPool')
        self._idle_actors = list(actors)
        self._future_to_actor = {}
        self._index_to_future = {}
        self._next_task_index = 0
        self._next_return_index = 0
        self._pending_submits = []

    def map(self, fn: Callable[['ray.actor.ActorHandle', V], Any], values: List[V]):
        if False:
            while True:
                i = 10
        'Apply the given function in parallel over the actors and values.\n\n        This returns an ordered iterator that will return results of the map\n        as they finish. Note that you must iterate over the iterator to force\n        the computation to finish.\n\n        Arguments:\n            fn: Function that takes (actor, value) as argument and\n                returns an ObjectRef computing the result over the value. The\n                actor will be considered busy until the ObjectRef completes.\n            values: List of values that fn(actor, value) should be\n                applied to.\n\n        Returns:\n            Iterator over results from applying fn to the actors and values.\n\n        Examples:\n            .. testcode::\n\n                import ray\n                from ray.util.actor_pool import ActorPool\n\n                @ray.remote\n                class Actor:\n                    def double(self, v):\n                        return 2 * v\n\n                a1, a2 = Actor.remote(), Actor.remote()\n                pool = ActorPool([a1, a2])\n                print(list(pool.map(lambda a, v: a.double.remote(v),\n                                    [1, 2, 3, 4])))\n\n            .. testoutput::\n\n                [2, 4, 6, 8]\n        '
        while self.has_next():
            try:
                self.get_next(timeout=0, ignore_if_timedout=True)
            except TimeoutError:
                pass
        for v in values:
            self.submit(fn, v)

        def get_generator():
            if False:
                while True:
                    i = 10
            while self.has_next():
                yield self.get_next()
        return get_generator()

    def map_unordered(self, fn: Callable[['ray.actor.ActorHandle', V], Any], values: List[V]):
        if False:
            i = 10
            return i + 15
        'Similar to map(), but returning an unordered iterator.\n\n        This returns an unordered iterator that will return results of the map\n        as they finish. This can be more efficient that map() if some results\n        take longer to compute than others.\n\n        Arguments:\n            fn: Function that takes (actor, value) as argument and\n                returns an ObjectRef computing the result over the value. The\n                actor will be considered busy until the ObjectRef completes.\n            values: List of values that fn(actor, value) should be\n                applied to.\n\n        Returns:\n            Iterator over results from applying fn to the actors and values.\n\n        Examples:\n            .. testcode::\n\n                import ray\n                from ray.util.actor_pool import ActorPool\n\n                @ray.remote\n                class Actor:\n                    def double(self, v):\n                        return 2 * v\n\n                a1, a2 = Actor.remote(), Actor.remote()\n                pool = ActorPool([a1, a2])\n                print(list(pool.map_unordered(lambda a, v: a.double.remote(v),\n                                              [1, 2, 3, 4])))\n\n            .. testoutput::\n                :options: +MOCK\n\n                [6, 8, 4, 2]\n        '
        while self.has_next():
            try:
                self.get_next_unordered(timeout=0)
            except TimeoutError:
                pass
        for v in values:
            self.submit(fn, v)

        def get_generator():
            if False:
                while True:
                    i = 10
            while self.has_next():
                yield self.get_next_unordered()
        return get_generator()

    def submit(self, fn, value):
        if False:
            return 10
        'Schedule a single task to run in the pool.\n\n        This has the same argument semantics as map(), but takes on a single\n        value instead of a list of values. The result can be retrieved using\n        get_next() / get_next_unordered().\n\n        Arguments:\n            fn: Function that takes (actor, value) as argument and\n                returns an ObjectRef computing the result over the value. The\n                actor will be considered busy until the ObjectRef completes.\n            value: Value to compute a result for.\n\n        Examples:\n            .. testcode::\n\n                import ray\n                from ray.util.actor_pool import ActorPool\n\n                @ray.remote\n                class Actor:\n                    def double(self, v):\n                        return 2 * v\n\n                a1, a2 = Actor.remote(), Actor.remote()\n                pool = ActorPool([a1, a2])\n                pool.submit(lambda a, v: a.double.remote(v), 1)\n                pool.submit(lambda a, v: a.double.remote(v), 2)\n                print(pool.get_next(), pool.get_next())\n\n            .. testoutput::\n\n                2 4\n        '
        if self._idle_actors:
            actor = self._idle_actors.pop()
            future = fn(actor, value)
            future_key = tuple(future) if isinstance(future, list) else future
            self._future_to_actor[future_key] = (self._next_task_index, actor)
            self._index_to_future[self._next_task_index] = future
            self._next_task_index += 1
        else:
            self._pending_submits.append((fn, value))

    def has_next(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns whether there are any pending results to return.\n\n        Returns:\n            True if there are any pending results not yet returned.\n\n        Examples:\n            .. testcode::\n\n                import ray\n                from ray.util.actor_pool import ActorPool\n\n                @ray.remote\n                class Actor:\n                    def double(self, v):\n                        return 2 * v\n\n                a1, a2 = Actor.remote(), Actor.remote()\n                pool = ActorPool([a1, a2])\n                pool.submit(lambda a, v: a.double.remote(v), 1)\n                print(pool.has_next())\n                print(pool.get_next())\n                print(pool.has_next())\n\n            .. testoutput::\n\n                True\n                2\n                False\n        '
        return bool(self._future_to_actor)

    def get_next(self, timeout=None, ignore_if_timedout=False):
        if False:
            return 10
        'Returns the next pending result in order.\n\n        This returns the next result produced by submit(), blocking for up to\n        the specified timeout until it is available.\n\n        Returns:\n            The next result.\n\n        Raises:\n            TimeoutError if the timeout is reached.\n\n        Examples:\n            .. testcode::\n\n                import ray\n                from ray.util.actor_pool import ActorPool\n\n                @ray.remote\n                class Actor:\n                    def double(self, v):\n                        return 2 * v\n\n                a1, a2 = Actor.remote(), Actor.remote()\n                pool = ActorPool([a1, a2])\n                pool.submit(lambda a, v: a.double.remote(v), 1)\n                print(pool.get_next())\n\n            .. testoutput::\n\n                2\n        '
        if not self.has_next():
            raise StopIteration('No more results to get')
        if self._next_return_index >= self._next_task_index:
            raise ValueError('It is not allowed to call get_next() after get_next_unordered().')
        future = self._index_to_future[self._next_return_index]
        timeout_msg = 'Timed out waiting for result'
        raise_timeout_after_ignore = False
        if timeout is not None:
            (res, _) = ray.wait([future], timeout=timeout)
            if not res:
                if not ignore_if_timedout:
                    raise TimeoutError(timeout_msg)
                else:
                    raise_timeout_after_ignore = True
        del self._index_to_future[self._next_return_index]
        self._next_return_index += 1
        future_key = tuple(future) if isinstance(future, list) else future
        (i, a) = self._future_to_actor.pop(future_key)
        self._return_actor(a)
        if raise_timeout_after_ignore:
            raise TimeoutError(timeout_msg + '. The task {} has been ignored.'.format(future))
        return ray.get(future)

    def get_next_unordered(self, timeout=None, ignore_if_timedout=False):
        if False:
            while True:
                i = 10
        'Returns any of the next pending results.\n\n        This returns some result produced by submit(), blocking for up to\n        the specified timeout until it is available. Unlike get_next(), the\n        results are not always returned in same order as submitted, which can\n        improve performance.\n\n        Returns:\n            The next result.\n\n        Raises:\n            TimeoutError if the timeout is reached.\n\n        Examples:\n            .. testcode::\n\n                import ray\n                from ray.util.actor_pool import ActorPool\n\n                @ray.remote\n                class Actor:\n                    def double(self, v):\n                        return 2 * v\n\n                a1, a2 = Actor.remote(), Actor.remote()\n                pool = ActorPool([a1, a2])\n                pool.submit(lambda a, v: a.double.remote(v), 1)\n                pool.submit(lambda a, v: a.double.remote(v), 2)\n                print(pool.get_next_unordered())\n                print(pool.get_next_unordered())\n\n            .. testoutput::\n                :options: +MOCK\n\n                4\n                2\n        '
        if not self.has_next():
            raise StopIteration('No more results to get')
        (res, _) = ray.wait(list(self._future_to_actor), num_returns=1, timeout=timeout)
        timeout_msg = 'Timed out waiting for result'
        raise_timeout_after_ignore = False
        if res:
            [future] = res
        elif not ignore_if_timedout:
            raise TimeoutError(timeout_msg)
        else:
            raise_timeout_after_ignore = True
        (i, a) = self._future_to_actor.pop(future)
        self._return_actor(a)
        del self._index_to_future[i]
        self._next_return_index = max(self._next_return_index, i + 1)
        if raise_timeout_after_ignore:
            raise TimeoutError(timeout_msg + '. The task {} has been ignored.'.format(future))
        return ray.get(future)

    def _return_actor(self, actor):
        if False:
            i = 10
            return i + 15
        self._idle_actors.append(actor)
        if self._pending_submits:
            self.submit(*self._pending_submits.pop(0))

    def has_free(self):
        if False:
            i = 10
            return i + 15
        'Returns whether there are any idle actors available.\n\n        Returns:\n            True if there are any idle actors and no pending submits.\n\n        Examples:\n            .. testcode::\n\n                import ray\n                from ray.util.actor_pool import ActorPool\n\n                @ray.remote\n                class Actor:\n                    def double(self, v):\n                        return 2 * v\n\n                a1 = Actor.remote()\n                pool = ActorPool([a1])\n                pool.submit(lambda a, v: a.double.remote(v), 1)\n                print(pool.has_free())\n                print(pool.get_next())\n                print(pool.has_free())\n\n            .. testoutput::\n\n                False\n                2\n                True\n        '
        return len(self._idle_actors) > 0 and len(self._pending_submits) == 0

    def pop_idle(self):
        if False:
            return 10
        'Removes an idle actor from the pool.\n\n        Returns:\n            An idle actor if one is available.\n            None if no actor was free to be removed.\n\n        Examples:\n            .. testcode::\n\n                import ray\n                from ray.util.actor_pool import ActorPool\n\n                @ray.remote\n                class Actor:\n                    def double(self, v):\n                        return 2 * v\n\n                a1 = Actor.remote()\n                pool = ActorPool([a1])\n                pool.submit(lambda a, v: a.double.remote(v), 1)\n                assert pool.pop_idle() is None\n                assert pool.get_next() == 2\n                assert pool.pop_idle() == a1\n\n        '
        if self.has_free():
            return self._idle_actors.pop()
        return None

    def push(self, actor):
        if False:
            i = 10
            return i + 15
        'Pushes a new actor into the current list of idle actors.\n\n        Examples:\n            .. testcode::\n\n                import ray\n                from ray.util.actor_pool import ActorPool\n\n                @ray.remote\n                class Actor:\n                    def double(self, v):\n                        return 2 * v\n\n                a1, a2 = Actor.remote(), Actor.remote()\n                pool = ActorPool([a1])\n                pool.push(a2)\n        '
        busy_actors = []
        if self._future_to_actor.values():
            (_, busy_actors) = zip(*self._future_to_actor.values())
        if actor in self._idle_actors or actor in busy_actors:
            raise ValueError('Actor already belongs to current ActorPool')
        else:
            self._return_actor(actor)