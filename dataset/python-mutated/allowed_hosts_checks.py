import logging
import sys
from typing import List
from connector_ops import utils

def get_connectors_missing_allowed_hosts() -> List[utils.Connector]:
    if False:
        for i in range(10):
            print('nop')
    connectors_missing_allowed_hosts: List[utils.Connector] = []
    changed_connectors = utils.get_changed_connectors(destination=False, third_party=False)
    for connector in changed_connectors:
        if connector.requires_allowed_hosts_check:
            missing = not connector_has_allowed_hosts(connector)
            if missing:
                connectors_missing_allowed_hosts.append(connector)
    return connectors_missing_allowed_hosts

def connector_has_allowed_hosts(connector: utils.Connector) -> bool:
    if False:
        print('Hello World!')
    return connector.allowed_hosts is not None

def check_allowed_hosts():
    if False:
        return 10
    connectors_missing_allowed_hosts = get_connectors_missing_allowed_hosts()
    if connectors_missing_allowed_hosts:
        logging.error(f'The following connectors must include allowedHosts: {connectors_missing_allowed_hosts}')
        sys.exit(1)
    else:
        sys.exit(0)