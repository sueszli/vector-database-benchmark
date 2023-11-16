from ..nn.fl_client import FLClient

def init_fl_context(client_id: int, server_addr='localhost:8980'):
    if False:
        print('Hello World!')
    'Initialize FL Context. Need to be called before calling any FL Client algorithms.\n    \n    :param client_id: An integer, should be in range of [1, total_party_number].\n    :param server_addr: FL Server address.\n    '
    FLClient.load_config()
    FLClient.set_client_id(client_id)
    FLClient.set_target(server_addr)
    FLClient.ensure_initialized()