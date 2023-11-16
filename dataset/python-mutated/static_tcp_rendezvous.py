import datetime
import logging
from typing import Tuple, cast, Optional
from torch.distributed import Store, TCPStore, PrefixStore
from torch.distributed.elastic.rendezvous import RendezvousHandler, RendezvousParameters
from torch.distributed.elastic.rendezvous.utils import parse_rendezvous_endpoint
log = logging.getLogger(__name__)
_default_timeout_seconds = 600

class StaticTCPRendezvous(RendezvousHandler):
    """
    Static rendezvous that is a wrapper around the TCPStore.

    Creates TCPStore based on the input parameters with the
    listener on the agent with group_rank=0
    """

    def __init__(self, master_addr: str, master_port: int, rank: int, world_size: int, run_id: str, timeout: int):
        if False:
            while True:
                i = 10
        self.master_addr = master_addr
        self.master_port = master_port
        self.rank = rank
        self.world_size = world_size
        self.run_id = run_id
        self.timeout = datetime.timedelta(seconds=timeout)
        self._store: Optional[Store] = None

    def get_backend(self) -> str:
        if False:
            print('Hello World!')
        return 'static'

    def next_rendezvous(self) -> Tuple[Store, int, int]:
        if False:
            i = 10
            return i + 15
        log.info('Creating TCPStore as the c10d::Store implementation')
        if not self._store:
            is_master = self.rank == 0
            self._store = TCPStore(self.master_addr, self.master_port, self.world_size, is_master, self.timeout, multi_tenant=True)
        store = PrefixStore(self.run_id, self._store)
        return (store, self.rank, self.world_size)

    def is_closed(self):
        if False:
            i = 10
            return i + 15
        return False

    def set_closed(self):
        if False:
            return 10
        pass

    def num_nodes_waiting(self):
        if False:
            return 10
        return 0

    def get_run_id(self) -> str:
        if False:
            i = 10
            return i + 15
        return self.run_id

    def shutdown(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return True

def create_rdzv_handler(params: RendezvousParameters) -> RendezvousHandler:
    if False:
        print('Hello World!')
    if 'rank' not in params.config:
        raise ValueError('rank is absent in RendezvousParameters.Try add --node-rank to the cmd request')
    endpoint = params.endpoint.strip()
    if not endpoint:
        raise ValueError('endpoint is absent in RendezvousParametersTry add --master-port and --master-addr to the cmd request')
    (master_addr, master_port) = parse_rendezvous_endpoint(endpoint, -1)
    if master_port == -1:
        raise ValueError(f'Port is absent in endpoint: {endpoint}. Try launching with --master-port')
    world_size = params.max_nodes
    rank = cast(int, params.config.get('rank'))
    run_id = params.run_id
    if 'timeout' in params.config:
        timeout = int(params.config['timeout'])
    else:
        timeout = _default_timeout_seconds
    return StaticTCPRendezvous(master_addr, master_port, rank, world_size, run_id, timeout)