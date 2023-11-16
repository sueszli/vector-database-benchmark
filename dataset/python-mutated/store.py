from datetime import timedelta
from typing import List

def get_all(store, rank: int, prefix: str, size: int):
    if False:
        print('Hello World!')
    "\n    Given a store and a prefix, the method goes through the array of keys\n    of the following format: ``{prefix}{idx}``, where idx is in a range\n    from 0 to size, and tries to retrieve the data.\n\n    The Rank0 process waits at the end to make sure all other processes\n    finished the procedure before exiting.\n\n    Usage\n\n    ::\n\n     values = get_all(store, 'torchelastic/data', 3)\n     value1 = values[0] # retrieves the data for key torchelastic/data0\n     value2 = values[1] # retrieves the data for key torchelastic/data1\n     value3 = values[2] # retrieves the data for key torchelastic/data2\n\n    "
    data_arr = []
    for idx in range(size):
        data = store.get(f'{prefix}{idx}')
        data_arr.append(data)
    store.set(f'{prefix}{rank}.FIN', b'FIN')
    if rank == 0:
        for node_rank in range(size):
            store.get(f'{prefix}{node_rank}.FIN')
    return data_arr

def synchronize(store, data: bytes, rank: int, world_size: int, key_prefix: str, barrier_timeout: float=300) -> List[bytes]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Synchronizes ``world_size`` agents between each other using the underlying c10d store.\n    The ``data`` will be available on each of the agents.\n\n    Note: The data on the path is not deleted, as a result there can be stale data if\n        you use the same key_prefix twice.\n    '
    store.set_timeout(timedelta(seconds=barrier_timeout))
    store.set(f'{key_prefix}{rank}', data)
    agent_data = get_all(store, rank, key_prefix, world_size)
    return agent_data

def barrier(store, rank: int, world_size: int, key_prefix: str, barrier_timeout: float=300) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    A global lock between agents.\n\n    Note: Since the data is not removed from the store, the barrier can be used\n        once per unique ``key_prefix``.\n    '
    data = f'{rank}'.encode()
    synchronize(store, data, rank, world_size, key_prefix, barrier_timeout)