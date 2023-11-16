from unittest.mock import Mock, patch
import pytest
import sshtunnel
from superset.extensions.ssh import SSHManagerFactory

def test_ssh_tunnel_timeout_setting() -> None:
    if False:
        return 10
    app = Mock()
    app.config = {'SSH_TUNNEL_MAX_RETRIES': 2, 'SSH_TUNNEL_LOCAL_BIND_ADDRESS': 'test', 'SSH_TUNNEL_TIMEOUT_SEC': 123.0, 'SSH_TUNNEL_PACKET_TIMEOUT_SEC': 321.0, 'SSH_TUNNEL_MANAGER_CLASS': 'superset.extensions.ssh.SSHManager'}
    factory = SSHManagerFactory()
    factory.init_app(app)
    assert sshtunnel.TUNNEL_TIMEOUT == 123.0
    assert sshtunnel.SSH_TIMEOUT == 321.0