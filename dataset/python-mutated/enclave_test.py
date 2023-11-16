import syft as sy
from syft.service.response import SyftError

def test_enclave_root_client_exception():
    if False:
        i = 10
        return i + 15
    enclave_node = sy.orchestra.launch(name='enclave_node', node_type=sy.NodeType.ENCLAVE, dev_mode=True, reset=True, local_db=True)
    res = enclave_node.login(email='info@openmined.org', password='changethis')
    assert isinstance(res, SyftError)