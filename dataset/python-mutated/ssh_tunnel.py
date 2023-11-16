from typing import Any
from superset.constants import PASSWORD_MASK
from superset.databases.ssh_tunnel.models import SSHTunnel

def mask_password_info(ssh_tunnel: dict[str, Any]) -> dict[str, Any]:
    if False:
        return 10
    if ssh_tunnel.pop('password', None) is not None:
        ssh_tunnel['password'] = PASSWORD_MASK
    if ssh_tunnel.pop('private_key', None) is not None:
        ssh_tunnel['private_key'] = PASSWORD_MASK
    if ssh_tunnel.pop('private_key_password', None) is not None:
        ssh_tunnel['private_key_password'] = PASSWORD_MASK
    return ssh_tunnel

def unmask_password_info(ssh_tunnel: dict[str, Any], model: SSHTunnel) -> dict[str, Any]:
    if False:
        while True:
            i = 10
    if ssh_tunnel.get('password') == PASSWORD_MASK:
        ssh_tunnel['password'] = model.password
    if ssh_tunnel.get('private_key') == PASSWORD_MASK:
        ssh_tunnel['private_key'] = model.private_key
    if ssh_tunnel.get('private_key_password') == PASSWORD_MASK:
        ssh_tunnel['private_key_password'] = model.private_key_password
    return ssh_tunnel