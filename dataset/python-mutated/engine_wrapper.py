"""
The module with helper mixin for executing functions remotely.

To be used as a piece of building a Ray-based engine.
"""
import asyncio
import ray

@ray.remote
def _deploy_ray_func(func, *args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Wrap `func` to ease calling it remotely.\n\n    Parameters\n    ----------\n    func : callable\n        A local function that we want to call remotely.\n    *args : iterable\n        Positional arguments to pass to `func` when calling remotely.\n    **kwargs : dict\n        Keyword arguments to pass to `func` when calling remotely.\n\n    Returns\n    -------\n    ray.ObjectRef or list\n        Ray identifier of the result being put to Plasma store.\n    '
    return func(*args, **kwargs)

class RayWrapper:
    """Mixin that provides means of running functions remotely and getting local results."""

    @classmethod
    def deploy(cls, func, f_args=None, f_kwargs=None, num_returns=1):
        if False:
            for i in range(10):
                print('nop')
        '\n        Run local `func` remotely.\n\n        Parameters\n        ----------\n        func : callable or ray.ObjectID\n            The function to perform.\n        f_args : list or tuple, optional\n            Positional arguments to pass to ``func``.\n        f_kwargs : dict, optional\n            Keyword arguments to pass to ``func``.\n        num_returns : int, default: 1\n            Amount of return values expected from `func`.\n\n        Returns\n        -------\n        ray.ObjectRef or list\n            Ray identifier of the result being put to Plasma store.\n        '
        args = [] if f_args is None else f_args
        kwargs = {} if f_kwargs is None else f_kwargs
        return _deploy_ray_func.options(num_returns=num_returns).remote(func, *args, **kwargs)

    @classmethod
    def materialize(cls, obj_id):
        if False:
            while True:
                i = 10
        '\n        Get the value of object from the Plasma store.\n\n        Parameters\n        ----------\n        obj_id : ray.ObjectID\n            Ray object identifier to get the value by.\n\n        Returns\n        -------\n        object\n            Whatever was identified by `obj_id`.\n        '
        return ray.get(obj_id)

    @classmethod
    def put(cls, data, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Store an object in the object store.\n\n        Parameters\n        ----------\n        data : object\n            The Python object to be stored.\n        **kwargs : dict\n            Additional keyword arguments.\n\n        Returns\n        -------\n        ray.ObjectID\n            Ray object identifier to get the value by.\n        '
        return ray.put(data, **kwargs)

    @classmethod
    def wait(cls, obj_ids, num_returns=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Wait on the objects without materializing them (blocking operation).\n\n        ``ray.wait`` assumes a list of unique object references: see\n        https://github.com/modin-project/modin/issues/5045\n\n        Parameters\n        ----------\n        obj_ids : list, scalar\n        num_returns : int, optional\n        '
        if not isinstance(obj_ids, list):
            obj_ids = [obj_ids]
        unique_ids = list(set(obj_ids))
        if num_returns is None:
            num_returns = len(unique_ids)
        ray.wait(unique_ids, num_returns=num_returns)

@ray.remote
class SignalActor:
    """
    Help synchronize across tasks and actors on cluster.

    For details see: https://docs.ray.io/en/latest/advanced.html?highlight=signalactor#multi-node-synchronization-using-an-actor

    Parameters
    ----------
    event_count : int
        Number of events required for synchronization.
    """

    def __init__(self, event_count: int):
        if False:
            i = 10
            return i + 15
        self.events = [asyncio.Event() for _ in range(event_count)]

    def send(self, event_idx: int):
        if False:
            print('Hello World!')
        '\n        Indicate that event with `event_idx` has occured.\n\n        Parameters\n        ----------\n        event_idx : int\n        '
        self.events[event_idx].set()

    async def wait(self, event_idx: int):
        """
        Wait until event with `event_idx` has occured.

        Parameters
        ----------
        event_idx : int
        """
        await self.events[event_idx].wait()

    def is_set(self, event_idx: int) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Check that event with `event_idx` had occured or not.\n\n        Parameters\n        ----------\n        event_idx : int\n\n        Returns\n        -------\n        bool\n        '
        return self.events[event_idx].is_set()