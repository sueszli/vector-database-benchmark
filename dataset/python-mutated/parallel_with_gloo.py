import time
from multiprocessing import Manager, Process
from paddle.base import core
from paddle.distributed.fleet.base.private_helper_function import wait_server_ready
__all__ = []
_global_gloo_ctx = None

def _start_kv_server(port, http_server_d, size):
    if False:
        i = 10
        return i + 15
    from paddle.distributed.fleet.utils.http_server import KVServer
    http_server = KVServer(int(port), size=size)
    http_server.start()
    wait_seconds = 3
    while http_server_d.get('running', False) or not http_server.should_stop():
        time.sleep(wait_seconds)
    http_server.stop()

def gloo_init_parallel_env(rank_id, rank_num, server_endpoint):
    if False:
        return 10
    '\n    Initialize parallel environment with gloo for cpu only.\n\n    Args:\n        - rank_id（int, required) - the index of current rank;\n        - rank_num (int, required) - the number of ranks in this parallel env;\n        - server_endpoint (str, required) - endpoint of server to init gloo context in ip:port format;\n\n    Returns:\n        None\n\n    Examples:\n        .. code-block:: python\n\n            >>> import paddle\n            >>> import multiprocessing\n            >>> from contextlib import closing\n            >>> import socket\n\n            >>> port_set = set()\n\n            >>> def find_free_port():\n            ...     def _free_port():\n            ...         with closing(socket.socket(socket.AF_INET,\n            ...             socket.SOCK_STREAM)) as s:\n            ...             s.bind((\'\', 0))\n            ...             return s.getsockname()[1]\n            ...     while True:\n            ...         port = _free_port()\n            ...         if port not in port_set:\n            ...             port_set.add(port)\n            ...             return port\n\n            >>> def test_gloo_init(id, rank_num, server_endpoint):\n            ...     paddle.distributed.gloo_init_parallel_env(\n            ...         id, rank_num, server_endpoint)\n\n            >>> def test_gloo_init_with_multiprocess(num_of_ranks):\n            ...     jobs = []\n            ...     server_endpoint = "127.0.0.1:%s" % (find_free_port())\n            ...     for id in range(num_of_ranks):\n            ...         p = multiprocessing.Process(\n            ...             target=test_gloo_init,\n            ...             args=(id, num_of_ranks, server_endpoint))\n            ...         jobs.append(p)\n            ...         p.start()\n            ...     for proc in jobs:\n            ...         proc.join()\n\n            >>> if __name__ == \'__main__\':\n            ...     # Arg: number of ranks (processes)\n            ...     test_gloo_init_with_multiprocess(2)\n    '
    assert (rank_num < 2) is False, 'rank_num should greater than or equal to 2 for parallel environment initialzation.'
    manager = Manager()
    http_server_status = manager.dict()
    http_server_status['running'] = False
    if rank_id == 0:
        size = {'_worker': rank_num}
        http_server_proc = Process(target=_start_kv_server, args=(int(server_endpoint.split(':')[1]), http_server_status, size))
        http_server_proc.daemon = True
        http_server_status['running'] = True
        http_server_proc.start()
    wait_server_ready([server_endpoint])
    gloo_strategy = core.GlooParallelStrategy()
    gloo_strategy.rank = rank_id
    gloo_strategy.rank_num = rank_num
    gloo_strategy.ip_address = server_endpoint.split(':')[0]
    gloo_strategy.ip_port = int(server_endpoint.split(':')[1])
    gloo_strategy.init_seconds = 3600
    gloo_strategy.run_seconds = 9999999
    global _global_gloo_ctx
    _global_gloo_ctx = core.GlooParallelContext(gloo_strategy)
    _global_gloo_ctx.init()
    if rank_id == 0:
        http_server_status['running'] = False
        http_server_proc.join()

def gloo_barrier():
    if False:
        print('Hello World!')
    '\n    Call barrier function with initialized gloo context.\n\n    Args:\n        None\n\n    Returns:\n        None\n\n    Examples:\n        .. code-block:: python\n\n            >>> import paddle\n            >>> import multiprocessing\n            >>> from contextlib import closing\n            >>> import socket\n\n            >>> port_set = set()\n\n            >>> def find_free_port():\n            ...     def _free_port():\n            ...         with closing(socket.socket(socket.AF_INET,\n            ...             socket.SOCK_STREAM)) as s:\n            ...             s.bind((\'\', 0))\n            ...             return s.getsockname()[1]\n            ...     while True:\n            ...         port = _free_port()\n            ...         if port not in port_set:\n            ...             port_set.add(port)\n            ...             return port\n\n            >>> def test_gloo_barrier(id, rank_num, server_endpoint):\n            ...     paddle.distributed.gloo_init_parallel_env(\n            ...         id, rank_num, server_endpoint)\n            ...     paddle.distributed.gloo_barrier()\n\n            >>> def test_gloo_barrier_with_multiprocess(num_of_ranks):\n            ...     jobs = []\n            ...     server_endpoint = "127.0.0.1:%s" % (find_free_port())\n            ...     for id in range(num_of_ranks):\n            ...         p = multiprocessing.Process(\n            ...             target=test_gloo_barrier,\n            ...             args=(id, num_of_ranks, server_endpoint))\n            ...         jobs.append(p)\n            ...         p.start()\n            ...     for proc in jobs:\n            ...         proc.join()\n\n            >>> if __name__ == \'__main__\':\n            ...     # Arg: number of ranks (processes)\n            ...     test_gloo_barrier_with_multiprocess(2)\n    '
    assert _global_gloo_ctx is not None, 'gloo context is not initialzed.'
    _global_gloo_ctx.barrier()

def gloo_release():
    if False:
        i = 10
        return i + 15
    '\n    Release the parallel environment initialized by gloo\n\n    Args:\n        None\n\n    Returns:\n        None\n\n    Examples:\n        .. code-block:: python\n\n            >>> import paddle\n            >>> import multiprocessing\n            >>> from contextlib import closing\n            >>> import socket\n\n            >>> port_set = set()\n\n            >>> def find_free_port():\n            ...     def _free_port():\n            ...         with closing(socket.socket(socket.AF_INET,\n            ...             socket.SOCK_STREAM)) as s:\n            ...             s.bind((\'\', 0))\n            ...             return s.getsockname()[1]\n            ...     while True:\n            ...         port = _free_port()\n            ...         if port not in port_set:\n            ...             port_set.add(port)\n            ...             return port\n\n            >>> def test_gloo_release(id, rank_num, server_endpoint):\n            ...     paddle.distributed.gloo_init_parallel_env(\n            ...         id, rank_num, server_endpoint)\n            ...     paddle.distributed.gloo_barrier()\n            ...     paddle.distributed.gloo_release()\n\n            >>> def test_gloo_release_with_multiprocess(num_of_ranks):\n            ...     jobs = []\n            ...     server_endpoint = "127.0.0.1:%s" % (find_free_port())\n            ...     for id in range(num_of_ranks):\n            ...         p = multiprocessing.Process(\n            ...             target=test_gloo_release,\n            ...             args=(id, num_of_ranks, server_endpoint))\n            ...         jobs.append(p)\n            ...         p.start()\n            ...     for proc in jobs:\n            ...         proc.join()\n\n            >>> if __name__ == \'__main__\':\n            ...     # Arg: number of ranks (processes)\n            ...     test_gloo_release_with_multiprocess(2)\n    '
    if _global_gloo_ctx is not None:
        _global_gloo_ctx.release()