"""
An execution module which can manipulate an f5 bigip via iControl REST
    :maturity:      develop
    :platform:      f5_bigip_11.6
"""
import salt.exceptions
import salt.utils.json
try:
    import requests
    import requests.exceptions
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False
__virtualname__ = 'bigip'

def __virtual__():
    if False:
        return 10
    '\n    Only return if requests is installed\n    '
    if HAS_LIBS:
        return __virtualname__
    return (False, 'The bigip execution module cannot be loaded: python requests library not available.')
BIG_IP_URL_BASE = 'https://{host}/mgmt/tm'

def _build_session(username, password, trans_label=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a session to be used when connecting to iControl REST.\n    '
    bigip = requests.session()
    bigip.auth = (username, password)
    bigip.verify = True
    bigip.headers.update({'Content-Type': 'application/json'})
    if trans_label:
        trans_id = __salt__['grains.get']('bigip_f5_trans:{label}'.format(label=trans_label))
        if trans_id:
            bigip.headers.update({'X-F5-REST-Coordination-Id': trans_id})
        else:
            bigip.headers.update({'X-F5-REST-Coordination-Id': None})
    return bigip

def _load_response(response):
    if False:
        for i in range(10):
            print('nop')
    '\n    Load the response from json data, return the dictionary or raw text\n    '
    try:
        data = salt.utils.json.loads(response.text)
    except ValueError:
        data = response.text
    ret = {'code': response.status_code, 'content': data}
    return ret

def _load_connection_error(hostname, error):
    if False:
        for i in range(10):
            print('nop')
    '\n    Format and Return a connection error\n    '
    ret = {'code': None, 'content': 'Error: Unable to connect to the bigip device: {host}\n{error}'.format(host=hostname, error=error)}
    return ret

def _loop_payload(params):
    if False:
        for i in range(10):
            print('nop')
    "\n    Pass in a dictionary of parameters, loop through them and build a payload containing,\n    parameters who's values are not None.\n    "
    payload = {}
    for (param, value) in params.items():
        if value is not None:
            payload[param] = value
    return payload

def _build_list(option_value, item_kind):
    if False:
        while True:
            i = 10
    '\n    pass in an option to check for a list of items, create a list of dictionary of items to set\n    for this option\n    '
    if option_value is not None:
        items = []
        if option_value == 'none':
            return items
        if not isinstance(option_value, list):
            values = option_value.split(',')
        else:
            values = option_value
        for value in values:
            if item_kind is None:
                items.append(value)
            else:
                items.append({'kind': item_kind, 'name': value})
        return items
    return None

def _determine_toggles(payload, toggles):
    if False:
        while True:
            i = 10
    "\n    BigIP can't make up its mind if it likes yes / no or true or false.\n    Figure out what it likes to hear without confusing the user.\n    "
    for (toggle, definition) in toggles.items():
        if definition['value'] is not None:
            if (definition['value'] is True or definition['value'] == 'yes') and definition['type'] == 'yes_no':
                payload[toggle] = 'yes'
            elif (definition['value'] is False or definition['value'] == 'no') and definition['type'] == 'yes_no':
                payload[toggle] = 'no'
            if (definition['value'] is True or definition['value'] == 'yes') and definition['type'] == 'true_false':
                payload[toggle] = True
            elif (definition['value'] is False or definition['value'] == 'no') and definition['type'] == 'true_false':
                payload[toggle] = False
    return payload

def _set_value(value):
    if False:
        return 10
    '\n    A function to detect if user is trying to pass a dictionary or list.  parse it and return a\n    dictionary list or a string\n    '
    if isinstance(value, bool) or isinstance(value, dict) or isinstance(value, list):
        return value
    if value.startswith('j{') and value.endswith('}j'):
        value = value.replace('j{', '{')
        value = value.replace('}j', '}')
        try:
            return salt.utils.json.loads(value)
        except Exception:
            raise salt.exceptions.CommandExecutionError
    if '|' in value and '\\|' not in value:
        values = value.split('|')
        items = []
        for value in values:
            items.append(_set_value(value))
        return items
    if ':' in value and '\\:' not in value:
        options = {}
        key_pairs = value.split(',')
        for key_pair in key_pairs:
            k = key_pair.split(':')[0]
            v = key_pair.split(':')[1]
            options[k] = v
        return options
    elif ',' in value and '\\,' not in value:
        value_items = value.split(',')
        return value_items
    else:
        if '\\|' in value:
            value = value.replace('\\|', '|')
        if '\\:' in value:
            value = value.replace('\\:', ':')
        if '\\,' in value:
            value = value.replace('\\,', ',')
        return value

def start_transaction(hostname, username, password, label):
    if False:
        print('Hello World!')
    "\n    A function to connect to a bigip device and start a new transaction.\n\n    hostname\n        The host/address of the bigip device\n    username\n        The iControl REST username\n    password\n        The iControl REST password\n    label\n        The name / alias for this transaction.  The actual transaction\n        id will be stored within a grain called ``bigip_f5_trans:<label>``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' bigip.start_transaction bigip admin admin my_transaction\n\n    "
    bigip_session = _build_session(username, password)
    payload = {}
    try:
        response = bigip_session.post(BIG_IP_URL_BASE.format(host=hostname) + '/transaction', data=salt.utils.json.dumps(payload))
    except requests.exceptions.ConnectionError as e:
        return _load_connection_error(hostname, e)
    data = _load_response(response)
    if data['code'] == 200:
        trans_id = data['content']['transId']
        __salt__['grains.setval']('bigip_f5_trans', {label: trans_id})
        return 'Transaction: {trans_id} - has successfully been stored in the grain: bigip_f5_trans:{label}'.format(trans_id=trans_id, label=label)
    else:
        return data

def list_transaction(hostname, username, password, label):
    if False:
        while True:
            i = 10
    "\n    A function to connect to a bigip device and list an existing transaction.\n\n    hostname\n        The host/address of the bigip device\n    username\n        The iControl REST username\n    password\n        The iControl REST password\n    label\n        the label of this transaction stored within the grain:\n        ``bigip_f5_trans:<label>``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' bigip.list_transaction bigip admin admin my_transaction\n\n    "
    bigip_session = _build_session(username, password)
    trans_id = __salt__['grains.get']('bigip_f5_trans:{label}'.format(label=label))
    if trans_id:
        try:
            response = bigip_session.get(BIG_IP_URL_BASE.format(host=hostname) + '/transaction/{trans_id}/commands'.format(trans_id=trans_id))
            return _load_response(response)
        except requests.exceptions.ConnectionError as e:
            return _load_connection_error(hostname, e)
    else:
        return 'Error: the label for this transaction was not defined as a grain.  Begin a new transaction using the bigip.start_transaction function'

def commit_transaction(hostname, username, password, label):
    if False:
        for i in range(10):
            print('nop')
    "\n    A function to connect to a bigip device and commit an existing transaction.\n\n    hostname\n        The host/address of the bigip device\n    username\n        The iControl REST username\n    password\n        The iControl REST password\n    label\n        the label of this transaction stored within the grain:\n        ``bigip_f5_trans:<label>``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' bigip.commit_transaction bigip admin admin my_transaction\n    "
    bigip_session = _build_session(username, password)
    trans_id = __salt__['grains.get']('bigip_f5_trans:{label}'.format(label=label))
    if trans_id:
        payload = {}
        payload['state'] = 'VALIDATING'
        try:
            response = bigip_session.patch(BIG_IP_URL_BASE.format(host=hostname) + '/transaction/{trans_id}'.format(trans_id=trans_id), data=salt.utils.json.dumps(payload))
            return _load_response(response)
        except requests.exceptions.ConnectionError as e:
            return _load_connection_error(hostname, e)
    else:
        return 'Error: the label for this transaction was not defined as a grain.  Begin a new transaction using the bigip.start_transaction function'

def delete_transaction(hostname, username, password, label):
    if False:
        return 10
    "\n    A function to connect to a bigip device and delete an existing transaction.\n\n    hostname\n        The host/address of the bigip device\n    username\n        The iControl REST username\n    password\n        The iControl REST password\n    label\n        The label of this transaction stored within the grain:\n        ``bigip_f5_trans:<label>``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' bigip.delete_transaction bigip admin admin my_transaction\n    "
    bigip_session = _build_session(username, password)
    trans_id = __salt__['grains.get']('bigip_f5_trans:{label}'.format(label=label))
    if trans_id:
        try:
            response = bigip_session.delete(BIG_IP_URL_BASE.format(host=hostname) + '/transaction/{trans_id}'.format(trans_id=trans_id))
            return _load_response(response)
        except requests.exceptions.ConnectionError as e:
            return _load_connection_error(hostname, e)
    else:
        return 'Error: the label for this transaction was not defined as a grain.  Begin a new transaction using the bigip.start_transaction function'

def list_node(hostname, username, password, name=None, trans_label=None):
    if False:
        return 10
    "\n    A function to connect to a bigip device and list all nodes or a specific node.\n\n\n    hostname\n        The host/address of the bigip device\n    username\n        The iControl REST username\n    password\n        The iControl REST password\n    name\n        The name of the node to list. If no name is specified than all nodes\n        will be listed.\n    trans_label\n        The label of the transaction stored within the grain:\n        ``bigip_f5_trans:<label>``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' bigip.list_node bigip admin admin my-node\n    "
    bigip_session = _build_session(username, password, trans_label)
    try:
        if name:
            response = bigip_session.get(BIG_IP_URL_BASE.format(host=hostname) + '/ltm/node/{name}'.format(name=name))
        else:
            response = bigip_session.get(BIG_IP_URL_BASE.format(host=hostname) + '/ltm/node')
    except requests.exceptions.ConnectionError as e:
        return _load_connection_error(hostname, e)
    return _load_response(response)

def create_node(hostname, username, password, name, address, trans_label=None):
    if False:
        print('Hello World!')
    "\n    A function to connect to a bigip device and create a node.\n\n    hostname\n        The host/address of the bigip device\n    username\n        The iControl REST username\n    password\n        The iControl REST password\n    name\n        The name of the node\n    address\n        The address of the node\n    trans_label\n        The label of the transaction stored within the grain:\n        ``bigip_f5_trans:<label>``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' bigip.create_node bigip admin admin 10.1.1.2\n    "
    bigip_session = _build_session(username, password, trans_label)
    payload = {}
    payload['name'] = name
    payload['address'] = address
    try:
        response = bigip_session.post(BIG_IP_URL_BASE.format(host=hostname) + '/ltm/node', data=salt.utils.json.dumps(payload))
    except requests.exceptions.ConnectionError as e:
        return _load_connection_error(hostname, e)
    return _load_response(response)

def modify_node(hostname, username, password, name, connection_limit=None, description=None, dynamic_ratio=None, logging=None, monitor=None, rate_limit=None, ratio=None, session=None, state=None, trans_label=None):
    if False:
        i = 10
        return i + 15
    "\n    A function to connect to a bigip device and modify an existing node.\n\n    hostname\n        The host/address of the bigip device\n    username\n        The iControl REST username\n    password\n        The iControl REST password\n    name\n        The name of the node to modify\n    connection_limit\n        [integer]\n    description\n        [string]\n    dynamic_ratio\n        [integer]\n    logging\n        [enabled | disabled]\n    monitor\n        [[name] | none | default]\n    rate_limit\n        [integer]\n    ratio\n        [integer]\n    session\n        [user-enabled | user-disabled]\n    state\n        [user-down | user-up ]\n    trans_label\n        The label of the transaction stored within the grain:\n        ``bigip_f5_trans:<label>``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' bigip.modify_node bigip admin admin 10.1.1.2 ratio=2 logging=enabled\n    "
    params = {'connection-limit': connection_limit, 'description': description, 'dynamic-ratio': dynamic_ratio, 'logging': logging, 'monitor': monitor, 'rate-limit': rate_limit, 'ratio': ratio, 'session': session, 'state': state}
    bigip_session = _build_session(username, password, trans_label)
    payload = _loop_payload(params)
    payload['name'] = name
    try:
        response = bigip_session.put(BIG_IP_URL_BASE.format(host=hostname) + '/ltm/node/{name}'.format(name=name), data=salt.utils.json.dumps(payload))
    except requests.exceptions.ConnectionError as e:
        return _load_connection_error(hostname, e)
    return _load_response(response)

def delete_node(hostname, username, password, name, trans_label=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    A function to connect to a bigip device and delete a specific node.\n\n    hostname\n        The host/address of the bigip device\n    username\n        The iControl REST username\n    password\n        The iControl REST password\n    name\n        The name of the node which will be deleted.\n    trans_label\n        The label of the transaction stored within the grain:\n        ``bigip_f5_trans:<label>``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' bigip.delete_node bigip admin admin my-node\n    "
    bigip_session = _build_session(username, password, trans_label)
    try:
        response = bigip_session.delete(BIG_IP_URL_BASE.format(host=hostname) + '/ltm/node/{name}'.format(name=name))
    except requests.exceptions.ConnectionError as e:
        return _load_connection_error(hostname, e)
    if _load_response(response) == '':
        return True
    else:
        return _load_response(response)

def list_pool(hostname, username, password, name=None):
    if False:
        print('Hello World!')
    "\n    A function to connect to a bigip device and list all pools or a specific pool.\n\n    hostname\n        The host/address of the bigip device\n    username\n        The iControl REST username\n    password\n        The iControl REST password\n    name\n        The name of the pool to list. If no name is specified then all pools\n        will be listed.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' bigip.list_pool bigip admin admin my-pool\n    "
    bigip_session = _build_session(username, password)
    try:
        if name:
            response = bigip_session.get(BIG_IP_URL_BASE.format(host=hostname) + '/ltm/pool/{name}/?expandSubcollections=true'.format(name=name))
        else:
            response = bigip_session.get(BIG_IP_URL_BASE.format(host=hostname) + '/ltm/pool')
    except requests.exceptions.ConnectionError as e:
        return _load_connection_error(hostname, e)
    return _load_response(response)

def create_pool(hostname, username, password, name, members=None, allow_nat=None, allow_snat=None, description=None, gateway_failsafe_device=None, ignore_persisted_weight=None, ip_tos_to_client=None, ip_tos_to_server=None, link_qos_to_client=None, link_qos_to_server=None, load_balancing_mode=None, min_active_members=None, min_up_members=None, min_up_members_action=None, min_up_members_checking=None, monitor=None, profiles=None, queue_depth_limit=None, queue_on_connection_limit=None, queue_time_limit=None, reselect_tries=None, service_down_action=None, slow_ramp_time=None):
    if False:
        i = 10
        return i + 15
    "\n    A function to connect to a bigip device and create a pool.\n\n    hostname\n        The host/address of the bigip device\n    username\n        The iControl REST username\n    password\n        The iControl REST password\n    name\n        The name of the pool to create.\n    members\n        List of comma delimited pool members to add to the pool.\n        i.e. 10.1.1.1:80,10.1.1.2:80,10.1.1.3:80\n    allow_nat\n        [yes | no]\n    allow_snat\n        [yes | no]\n    description\n        [string]\n    gateway_failsafe_device\n        [string]\n    ignore_persisted_weight\n        [enabled | disabled]\n    ip_tos_to_client\n        [pass-through | [integer]]\n    ip_tos_to_server\n        [pass-through | [integer]]\n    link_qos_to_client\n        [pass-through | [integer]]\n    link_qos_to_server\n        [pass-through | [integer]]\n    load_balancing_mode\n        [dynamic-ratio-member | dynamic-ratio-node |\n        fastest-app-response | fastest-node |\n        least-connections-members |\n        least-connections-node |\n        least-sessions |\n        observed-member | observed-node |\n        predictive-member | predictive-node |\n        ratio-least-connections-member |\n        ratio-least-connections-node |\n        ratio-member | ratio-node | ratio-session |\n        round-robin | weighted-least-connections-member |\n        weighted-least-connections-node]\n    min_active_members\n        [integer]\n    min_up_members\n        [integer]\n    min_up_members_action\n        [failover | reboot | restart-all]\n    min_up_members_checking\n        [enabled | disabled]\n    monitor\n        [name]\n    profiles\n        [none | profile_name]\n    queue_depth_limit\n        [integer]\n    queue_on_connection_limit\n        [enabled | disabled]\n    queue_time_limit\n        [integer]\n    reselect_tries\n        [integer]\n    service_down_action\n        [drop | none | reselect | reset]\n    slow_ramp_time\n        [integer]\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' bigip.create_pool bigip admin admin my-pool 10.1.1.1:80,10.1.1.2:80,10.1.1.3:80 monitor=http\n    "
    params = {'description': description, 'gateway-failsafe-device': gateway_failsafe_device, 'ignore-persisted-weight': ignore_persisted_weight, 'ip-tos-to-client': ip_tos_to_client, 'ip-tos-to-server': ip_tos_to_server, 'link-qos-to-client': link_qos_to_client, 'link-qos-to-server': link_qos_to_server, 'load-balancing-mode': load_balancing_mode, 'min-active-members': min_active_members, 'min-up-members': min_up_members, 'min-up-members-action': min_up_members_action, 'min-up-members-checking': min_up_members_checking, 'monitor': monitor, 'profiles': profiles, 'queue-on-connection-limit': queue_on_connection_limit, 'queue-depth-limit': queue_depth_limit, 'queue-time-limit': queue_time_limit, 'reselect-tries': reselect_tries, 'service-down-action': service_down_action, 'slow-ramp-time': slow_ramp_time}
    toggles = {'allow-nat': {'type': 'yes_no', 'value': allow_nat}, 'allow-snat': {'type': 'yes_no', 'value': allow_snat}}
    payload = _loop_payload(params)
    payload['name'] = name
    payload = _determine_toggles(payload, toggles)
    if members is not None:
        payload['members'] = _build_list(members, 'ltm:pool:members')
    bigip_session = _build_session(username, password)
    try:
        response = bigip_session.post(BIG_IP_URL_BASE.format(host=hostname) + '/ltm/pool', data=salt.utils.json.dumps(payload))
    except requests.exceptions.ConnectionError as e:
        return _load_connection_error(hostname, e)
    return _load_response(response)

def modify_pool(hostname, username, password, name, allow_nat=None, allow_snat=None, description=None, gateway_failsafe_device=None, ignore_persisted_weight=None, ip_tos_to_client=None, ip_tos_to_server=None, link_qos_to_client=None, link_qos_to_server=None, load_balancing_mode=None, min_active_members=None, min_up_members=None, min_up_members_action=None, min_up_members_checking=None, monitor=None, profiles=None, queue_depth_limit=None, queue_on_connection_limit=None, queue_time_limit=None, reselect_tries=None, service_down_action=None, slow_ramp_time=None):
    if False:
        print('Hello World!')
    "\n    A function to connect to a bigip device and modify an existing pool.\n\n    hostname\n        The host/address of the bigip device\n    username\n        The iControl REST username\n    password\n        The iControl REST password\n    name\n        The name of the pool to modify.\n    allow_nat\n        [yes | no]\n    allow_snat\n        [yes | no]\n    description\n        [string]\n    gateway_failsafe_device\n        [string]\n    ignore_persisted_weight\n        [yes | no]\n    ip_tos_to_client\n        [pass-through | [integer]]\n    ip_tos_to_server\n        [pass-through | [integer]]\n    link_qos_to_client\n        [pass-through | [integer]]\n    link_qos_to_server\n        [pass-through | [integer]]\n    load_balancing_mode\n        [dynamic-ratio-member | dynamic-ratio-node |\n        fastest-app-response | fastest-node |\n        least-connections-members |\n        least-connections-node |\n        least-sessions |\n        observed-member | observed-node |\n        predictive-member | predictive-node |\n        ratio-least-connections-member |\n        ratio-least-connections-node |\n        ratio-member | ratio-node | ratio-session |\n        round-robin | weighted-least-connections-member |\n        weighted-least-connections-node]\n    min_active_members\n        [integer]\n    min_up_members\n        [integer]\n    min_up_members_action\n        [failover | reboot | restart-all]\n    min_up_members_checking\n        [enabled | disabled]\n    monitor\n        [name]\n    profiles\n        [none | profile_name]\n    queue_on_connection_limit\n        [enabled | disabled]\n    queue_depth_limit\n        [integer]\n    queue_time_limit\n        [integer]\n    reselect_tries\n        [integer]\n    service_down_action\n        [drop | none | reselect | reset]\n    slow_ramp_time\n        [integer]\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' bigip.modify_pool bigip admin admin my-pool 10.1.1.1:80,10.1.1.2:80,10.1.1.3:80 min_active_members=1\n    "
    params = {'description': description, 'gateway-failsafe-device': gateway_failsafe_device, 'ignore-persisted-weight': ignore_persisted_weight, 'ip-tos-to-client': ip_tos_to_client, 'ip-tos-to-server': ip_tos_to_server, 'link-qos-to-client': link_qos_to_client, 'link-qos-to-server': link_qos_to_server, 'load-balancing-mode': load_balancing_mode, 'min-active-members': min_active_members, 'min-up-members': min_up_members, 'min-up_members-action': min_up_members_action, 'min-up-members-checking': min_up_members_checking, 'monitor': monitor, 'profiles': profiles, 'queue-on-connection-limit': queue_on_connection_limit, 'queue-depth-limit': queue_depth_limit, 'queue-time-limit': queue_time_limit, 'reselect-tries': reselect_tries, 'service-down-action': service_down_action, 'slow-ramp-time': slow_ramp_time}
    toggles = {'allow-nat': {'type': 'yes_no', 'value': allow_nat}, 'allow-snat': {'type': 'yes_no', 'value': allow_snat}}
    payload = _loop_payload(params)
    payload['name'] = name
    payload = _determine_toggles(payload, toggles)
    bigip_session = _build_session(username, password)
    try:
        response = bigip_session.put(BIG_IP_URL_BASE.format(host=hostname) + '/ltm/pool/{name}'.format(name=name), data=salt.utils.json.dumps(payload))
    except requests.exceptions.ConnectionError as e:
        return _load_connection_error(hostname, e)
    return _load_response(response)

def delete_pool(hostname, username, password, name):
    if False:
        while True:
            i = 10
    "\n    A function to connect to a bigip device and delete a specific pool.\n\n    hostname\n        The host/address of the bigip device\n    username\n        The iControl REST username\n    password\n        The iControl REST password\n    name\n        The name of the pool which will be deleted\n\n    CLI Example\n\n    .. code-block:: bash\n\n        salt '*' bigip.delete_node bigip admin admin my-pool\n    "
    bigip_session = _build_session(username, password)
    try:
        response = bigip_session.delete(BIG_IP_URL_BASE.format(host=hostname) + '/ltm/pool/{name}'.format(name=name))
    except requests.exceptions.ConnectionError as e:
        return _load_connection_error(hostname, e)
    if _load_response(response) == '':
        return True
    else:
        return _load_response(response)

def replace_pool_members(hostname, username, password, name, members):
    if False:
        i = 10
        return i + 15
    "\n    A function to connect to a bigip device and replace members of an existing pool with new members.\n\n    hostname\n        The host/address of the bigip device\n    username\n        The iControl REST username\n    password\n        The iControl REST password\n    name\n        The name of the pool to modify\n    members\n        List of comma delimited pool members to replace existing members with.\n        i.e. 10.1.1.1:80,10.1.1.2:80,10.1.1.3:80\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' bigip.replace_pool_members bigip admin admin my-pool 10.2.2.1:80,10.2.2.2:80,10.2.2.3:80\n    "
    payload = {}
    payload['name'] = name
    if members is not None:
        if isinstance(members, str):
            members = members.split(',')
        pool_members = []
        for member in members:
            if isinstance(member, dict):
                if 'member_state' in member.keys():
                    member['state'] = member.pop('member_state')
                for key in member:
                    new_key = key.replace('_', '-')
                    member[new_key] = member.pop(key)
                pool_members.append(member)
            else:
                pool_members.append({'name': member, 'address': member.split(':')[0]})
        payload['members'] = pool_members
    bigip_session = _build_session(username, password)
    try:
        response = bigip_session.put(BIG_IP_URL_BASE.format(host=hostname) + '/ltm/pool/{name}'.format(name=name), data=salt.utils.json.dumps(payload))
    except requests.exceptions.ConnectionError as e:
        return _load_connection_error(hostname, e)
    return _load_response(response)

def add_pool_member(hostname, username, password, name, member):
    if False:
        return 10
    "\n    A function to connect to a bigip device and add a new member to an existing pool.\n\n    hostname\n        The host/address of the bigip device\n    username\n        The iControl REST username\n    password\n        The iControl REST password\n    name\n        The name of the pool to modify\n    member\n        The name of the member to add\n        i.e. 10.1.1.2:80\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' bigip.add_pool_members bigip admin admin my-pool 10.2.2.1:80\n    "
    if isinstance(member, dict):
        if 'member_state' in member.keys():
            member['state'] = member.pop('member_state')
        for key in member:
            new_key = key.replace('_', '-')
            member[new_key] = member.pop(key)
        payload = member
    else:
        payload = {'name': member, 'address': member.split(':')[0]}
    bigip_session = _build_session(username, password)
    try:
        response = bigip_session.post(BIG_IP_URL_BASE.format(host=hostname) + '/ltm/pool/{name}/members'.format(name=name), data=salt.utils.json.dumps(payload))
    except requests.exceptions.ConnectionError as e:
        return _load_connection_error(hostname, e)
    return _load_response(response)

def modify_pool_member(hostname, username, password, name, member, connection_limit=None, description=None, dynamic_ratio=None, inherit_profile=None, logging=None, monitor=None, priority_group=None, profiles=None, rate_limit=None, ratio=None, session=None, state=None):
    if False:
        return 10
    "\n    A function to connect to a bigip device and modify an existing member of a pool.\n\n    hostname\n        The host/address of the bigip device\n    username\n        The iControl REST username\n    password\n        The iControl REST password\n    name\n        The name of the pool to modify\n    member\n        The name of the member to modify i.e. 10.1.1.2:80\n    connection_limit\n        [integer]\n    description\n        [string]\n    dynamic_ratio\n        [integer]\n    inherit_profile\n        [enabled | disabled]\n    logging\n        [enabled | disabled]\n    monitor\n        [name]\n    priority_group\n        [integer]\n    profiles\n        [none | profile_name]\n    rate_limit\n        [integer]\n    ratio\n        [integer]\n    session\n        [user-enabled | user-disabled]\n    state\n        [ user-up | user-down ]\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' bigip.modify_pool_member bigip admin admin my-pool 10.2.2.1:80 state=use-down session=user-disabled\n    "
    params = {'connection-limit': connection_limit, 'description': description, 'dynamic-ratio': dynamic_ratio, 'inherit-profile': inherit_profile, 'logging': logging, 'monitor': monitor, 'priority-group': priority_group, 'profiles': profiles, 'rate-limit': rate_limit, 'ratio': ratio, 'session': session, 'state': state}
    bigip_session = _build_session(username, password)
    payload = _loop_payload(params)
    try:
        response = bigip_session.put(BIG_IP_URL_BASE.format(host=hostname) + '/ltm/pool/{name}/members/{member}'.format(name=name, member=member), data=salt.utils.json.dumps(payload))
    except requests.exceptions.ConnectionError as e:
        return _load_connection_error(hostname, e)
    return _load_response(response)

def delete_pool_member(hostname, username, password, name, member):
    if False:
        while True:
            i = 10
    "\n    A function to connect to a bigip device and delete a specific pool.\n\n    hostname\n        The host/address of the bigip device\n    username\n        The iControl REST username\n    password\n        The iControl REST password\n    name\n        The name of the pool to modify\n    member\n        The name of the pool member to delete\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' bigip.delete_pool_member bigip admin admin my-pool 10.2.2.2:80\n    "
    bigip_session = _build_session(username, password)
    try:
        response = bigip_session.delete(BIG_IP_URL_BASE.format(host=hostname) + '/ltm/pool/{name}/members/{member}'.format(name=name, member=member))
    except requests.exceptions.ConnectionError as e:
        return _load_connection_error(hostname, e)
    if _load_response(response) == '':
        return True
    else:
        return _load_response(response)

def list_virtual(hostname, username, password, name=None):
    if False:
        return 10
    "\n    A function to connect to a bigip device and list all virtuals or a specific virtual.\n\n    hostname\n        The host/address of the bigip device\n    username\n        The iControl REST username\n    password\n        The iControl REST password\n    name\n        The name of the virtual to list. If no name is specified than all\n        virtuals will be listed.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' bigip.list_virtual bigip admin admin my-virtual\n    "
    bigip_session = _build_session(username, password)
    try:
        if name:
            response = bigip_session.get(BIG_IP_URL_BASE.format(host=hostname) + '/ltm/virtual/{name}/?expandSubcollections=true'.format(name=name))
        else:
            response = bigip_session.get(BIG_IP_URL_BASE.format(host=hostname) + '/ltm/virtual')
    except requests.exceptions.ConnectionError as e:
        return _load_connection_error(hostname, e)
    return _load_response(response)

def create_virtual(hostname, username, password, name, destination, pool=None, address_status=None, auto_lasthop=None, bwc_policy=None, cmp_enabled=None, connection_limit=None, dhcp_relay=None, description=None, fallback_persistence=None, flow_eviction_policy=None, gtm_score=None, ip_forward=None, ip_protocol=None, internal=None, twelve_forward=None, last_hop_pool=None, mask=None, mirror=None, nat64=None, persist=None, profiles=None, policies=None, rate_class=None, rate_limit=None, rate_limit_mode=None, rate_limit_dst=None, rate_limit_src=None, rules=None, related_rules=None, reject=None, source=None, source_address_translation=None, source_port=None, state=None, traffic_classes=None, translate_address=None, translate_port=None, vlans=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    A function to connect to a bigip device and create a virtual server.\n\n    hostname\n        The host/address of the bigip device\n    username\n        The iControl REST username\n    password\n        The iControl REST password\n    name\n        The name of the virtual to create\n    destination\n        [ [virtual_address_name:port] | [ipv4:port] | [ipv6.port] ]\n    pool\n        [ [pool_name] | none]\n    address_status\n        [yes | no]\n    auto_lasthop\n        [default | enabled | disabled ]\n    bwc_policy\n        [none] | string]\n    cmp_enabled\n        [yes | no]\n    dhcp_relay\n        [yes | no]\n    connection_limit\n        [integer]\n    description\n        [string]\n    state\n        [disabled | enabled]\n    fallback_persistence\n        [none | [profile name] ]\n    flow_eviction_policy\n        [none | [eviction policy name] ]\n    gtm_score\n        [integer]\n    ip_forward\n        [yes | no]\n    ip_protocol\n        [any | protocol]\n    internal\n        [yes | no]\n    twelve_forward\n        (12-forward)\n        [yes | no]\n    last_hop-pool\n        [ [pool_name] | none]\n    mask\n        { [ipv4] | [ipv6] }\n    mirror\n        { [disabled | enabled | none] }\n    nat64\n        [enabled | disabled]\n    persist\n        [none | profile1,profile2,profile3 ... ]\n    profiles\n        [none | default | profile1,profile2,profile3 ... ]\n    policies\n        [none | default | policy1,policy2,policy3 ... ]\n    rate_class\n        [name]\n    rate_limit\n        [integer]\n    rate_limit_mode\n        [destination | object | object-destination |\n        object-source | object-source-destination |\n        source | source-destination]\n    rate_limit_dst\n        [integer]\n    rate_limitçsrc\n        [integer]\n    rules\n        [none | [rule_one,rule_two ...] ]\n    related_rules\n        [none | [rule_one,rule_two ...] ]\n    reject\n        [yes | no]\n    source\n        { [ipv4[/prefixlen]] | [ipv6[/prefixlen]] }\n    source_address_translation\n        [none | snat:pool_name | lsn | automap ]\n    source_port\n        [change | preserve | preserve-strict]\n    state\n        [enabled | disabled]\n    traffic_classes\n        [none | default | class_one,class_two ... ]\n    translate_address\n        [enabled | disabled]\n    translate_port\n        [enabled | disabled]\n    vlans\n        [none | default | [enabled|disabled]:vlan1,vlan2,vlan3 ... ]\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' bigip.create_virtual bigip admin admin my-virtual-3 26.2.2.5:80 \\\n            pool=my-http-pool-http profiles=http,tcp\n\n        salt '*' bigip.create_virtual bigip admin admin my-virtual-3 43.2.2.5:80 \\\n            pool=test-http-pool-http profiles=http,websecurity persist=cookie,hash \\\n            policies=asm_auto_l7_policy__http-virtual \\\n            rules=_sys_APM_ExchangeSupport_helper,_sys_https_redirect \\\n            related_rules=_sys_APM_activesync,_sys_APM_ExchangeSupport_helper \\\n            source_address_translation=snat:my-snat-pool \\\n            translate_address=enabled translate_port=enabled \\\n            traffic_classes=my-class,other-class \\\n            vlans=enabled:external,internal\n\n    "
    params = {'pool': pool, 'auto-lasthop': auto_lasthop, 'bwc-policy': bwc_policy, 'connection-limit': connection_limit, 'description': description, 'fallback-persistence': fallback_persistence, 'flow-eviction-policy': flow_eviction_policy, 'gtm-score': gtm_score, 'ip-protocol': ip_protocol, 'last-hop-pool': last_hop_pool, 'mask': mask, 'mirror': mirror, 'nat64': nat64, 'persist': persist, 'rate-class': rate_class, 'rate-limit': rate_limit, 'rate-limit-mode': rate_limit_mode, 'rate-limit-dst': rate_limit_dst, 'rate-limit-src': rate_limit_src, 'source': source, 'source-port': source_port, 'translate-address': translate_address, 'translate-port': translate_port}
    toggles = {'address-status': {'type': 'yes_no', 'value': address_status}, 'cmp-enabled': {'type': 'yes_no', 'value': cmp_enabled}, 'dhcp-relay': {'type': 'true_false', 'value': dhcp_relay}, 'reject': {'type': 'true_false', 'value': reject}, '12-forward': {'type': 'true_false', 'value': twelve_forward}, 'internal': {'type': 'true_false', 'value': internal}, 'ip-forward': {'type': 'true_false', 'value': ip_forward}}
    bigip_session = _build_session(username, password)
    payload = _loop_payload(params)
    payload['name'] = name
    payload['destination'] = destination
    payload = _determine_toggles(payload, toggles)
    if profiles is not None:
        payload['profiles'] = _build_list(profiles, 'ltm:virtual:profile')
    if persist is not None:
        payload['persist'] = _build_list(persist, 'ltm:virtual:persist')
    if policies is not None:
        payload['policies'] = _build_list(policies, 'ltm:virtual:policy')
    if rules is not None:
        payload['rules'] = _build_list(rules, None)
    if related_rules is not None:
        payload['related-rules'] = _build_list(related_rules, None)
    if source_address_translation is not None:
        if isinstance(source_address_translation, dict):
            payload['source-address-translation'] = source_address_translation
        elif source_address_translation == 'none':
            payload['source-address-translation'] = {'pool': 'none', 'type': 'none'}
        elif source_address_translation == 'automap':
            payload['source-address-translation'] = {'pool': 'none', 'type': 'automap'}
        elif source_address_translation == 'lsn':
            payload['source-address-translation'] = {'pool': 'none', 'type': 'lsn'}
        elif source_address_translation.startswith('snat'):
            snat_pool = source_address_translation.split(':')[1]
            payload['source-address-translation'] = {'pool': snat_pool, 'type': 'snat'}
    if traffic_classes is not None:
        payload['traffic-classes'] = _build_list(traffic_classes, None)
    if vlans is not None:
        if isinstance(vlans, dict):
            try:
                payload['vlans'] = vlans['vlan_ids']
                if vlans['enabled']:
                    payload['vlans-enabled'] = True
                elif vlans['disabled']:
                    payload['vlans-disabled'] = True
            except Exception:
                return 'Error: Unable to Parse vlans dictionary: \n\tvlans={vlans}'.format(vlans=vlans)
        elif vlans == 'none':
            payload['vlans'] = 'none'
        elif vlans == 'default':
            payload['vlans'] = 'default'
        elif isinstance(vlans, str) and (vlans.startswith('enabled') or vlans.startswith('disabled')):
            try:
                vlans_setting = vlans.split(':')[0]
                payload['vlans'] = vlans.split(':')[1].split(',')
                if vlans_setting == 'disabled':
                    payload['vlans-disabled'] = True
                elif vlans_setting == 'enabled':
                    payload['vlans-enabled'] = True
            except Exception:
                return 'Error: Unable to Parse vlans option: \n\tvlans={vlans}'.format(vlans=vlans)
        else:
            return 'Error: vlans must be a dictionary or string.'
    if state is not None:
        if state == 'enabled':
            payload['enabled'] = True
        elif state == 'disabled':
            payload['disabled'] = True
    try:
        response = bigip_session.post(BIG_IP_URL_BASE.format(host=hostname) + '/ltm/virtual', data=salt.utils.json.dumps(payload))
    except requests.exceptions.ConnectionError as e:
        return _load_connection_error(hostname, e)
    return _load_response(response)

def modify_virtual(hostname, username, password, name, destination=None, pool=None, address_status=None, auto_lasthop=None, bwc_policy=None, cmp_enabled=None, connection_limit=None, dhcp_relay=None, description=None, fallback_persistence=None, flow_eviction_policy=None, gtm_score=None, ip_forward=None, ip_protocol=None, internal=None, twelve_forward=None, last_hop_pool=None, mask=None, mirror=None, nat64=None, persist=None, profiles=None, policies=None, rate_class=None, rate_limit=None, rate_limit_mode=None, rate_limit_dst=None, rate_limit_src=None, rules=None, related_rules=None, reject=None, source=None, source_address_translation=None, source_port=None, state=None, traffic_classes=None, translate_address=None, translate_port=None, vlans=None):
    if False:
        i = 10
        return i + 15
    "\n    A function to connect to a bigip device and modify an existing virtual server.\n\n    hostname\n        The host/address of the bigip device\n    username\n        The iControl REST username\n    password\n        The iControl REST password\n    name\n        The name of the virtual to modify\n    destination\n        [ [virtual_address_name:port] | [ipv4:port] | [ipv6.port] ]\n    pool\n        [ [pool_name] | none]\n    address_status\n        [yes | no]\n    auto_lasthop\n        [default | enabled | disabled ]\n    bwc_policy\n        [none] | string]\n    cmp_enabled\n        [yes | no]\n    dhcp_relay\n        [yes | no}\n    connection_limit\n        [integer]\n    description\n        [string]\n    state\n        [disabled | enabled]\n    fallback_persistence\n        [none | [profile name] ]\n    flow_eviction_policy\n        [none | [eviction policy name] ]\n    gtm_score\n        [integer]\n    ip_forward\n        [yes | no]\n    ip_protocol\n        [any | protocol]\n    internal\n        [yes | no]\n    twelve_forward\n        (12-forward)\n        [yes | no]\n    last_hop-pool\n        [ [pool_name] | none]\n    mask\n        { [ipv4] | [ipv6] }\n    mirror\n        { [disabled | enabled | none] }\n    nat64\n        [enabled | disabled]\n    persist\n        [none | profile1,profile2,profile3 ... ]\n    profiles\n        [none | default | profile1,profile2,profile3 ... ]\n    policies\n        [none | default | policy1,policy2,policy3 ... ]\n    rate_class\n        [name]\n    rate_limit\n        [integer]\n    rate_limitr_mode\n        [destination | object | object-destination |\n        object-source | object-source-destination |\n        source | source-destination]\n    rate_limit_dst\n        [integer]\n    rate_limit_src\n        [integer]\n    rules\n        [none | [rule_one,rule_two ...] ]\n    related_rules\n        [none | [rule_one,rule_two ...] ]\n    reject\n        [yes | no]\n    source\n        { [ipv4[/prefixlen]] | [ipv6[/prefixlen]] }\n    source_address_translation\n        [none | snat:pool_name | lsn | automap ]\n    source_port\n        [change | preserve | preserve-strict]\n    state\n        [enabled | disable]\n    traffic_classes\n        [none | default | class_one,class_two ... ]\n    translate_address\n        [enabled | disabled]\n    translate_port\n        [enabled | disabled]\n    vlans\n        [none | default | [enabled|disabled]:vlan1,vlan2,vlan3 ... ]\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' bigip.modify_virtual bigip admin admin my-virtual source_address_translation=none\n        salt '*' bigip.modify_virtual bigip admin admin my-virtual rules=my-rule,my-other-rule\n    "
    params = {'destination': destination, 'pool': pool, 'auto-lasthop': auto_lasthop, 'bwc-policy': bwc_policy, 'connection-limit': connection_limit, 'description': description, 'fallback-persistence': fallback_persistence, 'flow-eviction-policy': flow_eviction_policy, 'gtm-score': gtm_score, 'ip-protocol': ip_protocol, 'last-hop-pool': last_hop_pool, 'mask': mask, 'mirror': mirror, 'nat64': nat64, 'persist': persist, 'rate-class': rate_class, 'rate-limit': rate_limit, 'rate-limit-mode': rate_limit_mode, 'rate-limit-dst': rate_limit_dst, 'rate-limit-src': rate_limit_src, 'source': source, 'source-port': source_port, 'translate-address': translate_address, 'translate-port': translate_port}
    toggles = {'address-status': {'type': 'yes_no', 'value': address_status}, 'cmp-enabled': {'type': 'yes_no', 'value': cmp_enabled}, 'dhcp-relay': {'type': 'true_false', 'value': dhcp_relay}, 'reject': {'type': 'true_false', 'value': reject}, '12-forward': {'type': 'true_false', 'value': twelve_forward}, 'internal': {'type': 'true_false', 'value': internal}, 'ip-forward': {'type': 'true_false', 'value': ip_forward}}
    bigip_session = _build_session(username, password)
    payload = _loop_payload(params)
    payload['name'] = name
    payload = _determine_toggles(payload, toggles)
    if profiles is not None:
        payload['profiles'] = _build_list(profiles, 'ltm:virtual:profile')
    if persist is not None:
        payload['persist'] = _build_list(persist, 'ltm:virtual:persist')
    if policies is not None:
        payload['policies'] = _build_list(policies, 'ltm:virtual:policy')
    if rules is not None:
        payload['rules'] = _build_list(rules, None)
    if related_rules is not None:
        payload['related-rules'] = _build_list(related_rules, None)
    if source_address_translation is not None:
        if source_address_translation == 'none':
            payload['source-address-translation'] = {'pool': 'none', 'type': 'none'}
        elif source_address_translation == 'automap':
            payload['source-address-translation'] = {'pool': 'none', 'type': 'automap'}
        elif source_address_translation == 'lsn':
            payload['source-address-translation'] = {'pool': 'none', 'type': 'lsn'}
        elif source_address_translation.startswith('snat'):
            snat_pool = source_address_translation.split(':')[1]
            payload['source-address-translation'] = {'pool': snat_pool, 'type': 'snat'}
    if traffic_classes is not None:
        payload['traffic-classes'] = _build_list(traffic_classes, None)
    if vlans is not None:
        if isinstance(vlans, dict):
            try:
                payload['vlans'] = vlans['vlan_ids']
                if vlans['enabled']:
                    payload['vlans-enabled'] = True
                elif vlans['disabled']:
                    payload['vlans-disabled'] = True
            except Exception:
                return 'Error: Unable to Parse vlans dictionary: \n\tvlans={vlans}'.format(vlans=vlans)
        elif vlans == 'none':
            payload['vlans'] = 'none'
        elif vlans == 'default':
            payload['vlans'] = 'default'
        elif vlans.startswith('enabled') or vlans.startswith('disabled'):
            try:
                vlans_setting = vlans.split(':')[0]
                payload['vlans'] = vlans.split(':')[1].split(',')
                if vlans_setting == 'disabled':
                    payload['vlans-disabled'] = True
                elif vlans_setting == 'enabled':
                    payload['vlans-enabled'] = True
            except Exception:
                return 'Error: Unable to Parse vlans option: \n\tvlans={vlans}'.format(vlans=vlans)
    if state is not None:
        if state == 'enabled':
            payload['enabled'] = True
        elif state == 'disabled':
            payload['disabled'] = True
    try:
        response = bigip_session.put(BIG_IP_URL_BASE.format(host=hostname) + '/ltm/virtual/{name}'.format(name=name), data=salt.utils.json.dumps(payload))
    except requests.exceptions.ConnectionError as e:
        return _load_connection_error(hostname, e)
    return _load_response(response)

def delete_virtual(hostname, username, password, name):
    if False:
        while True:
            i = 10
    "\n    A function to connect to a bigip device and delete a specific virtual.\n\n    hostname\n        The host/address of the bigip device\n    username\n        The iControl REST username\n    password\n        The iControl REST password\n    name\n        The name of the virtual to delete\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' bigip.delete_virtual bigip admin admin my-virtual\n    "
    bigip_session = _build_session(username, password)
    try:
        response = bigip_session.delete(BIG_IP_URL_BASE.format(host=hostname) + '/ltm/virtual/{name}'.format(name=name))
    except requests.exceptions.ConnectionError as e:
        return _load_connection_error(hostname, e)
    if _load_response(response) == '':
        return True
    else:
        return _load_response(response)

def list_monitor(hostname, username, password, monitor_type, name=None):
    if False:
        while True:
            i = 10
    "\n    A function to connect to a bigip device and list an existing monitor.  If no name is provided than all\n    monitors of the specified type will be listed.\n\n    hostname\n        The host/address of the bigip device\n    username\n        The iControl REST username\n    password\n        The iControl REST password\n    monitor_type\n        The type of monitor(s) to list\n    name\n        The name of the monitor to list\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' bigip.list_monitor bigip admin admin http my-http-monitor\n\n    "
    bigip_session = _build_session(username, password)
    try:
        if name:
            response = bigip_session.get(BIG_IP_URL_BASE.format(host=hostname) + '/ltm/monitor/{type}/{name}?expandSubcollections=true'.format(type=monitor_type, name=name))
        else:
            response = bigip_session.get(BIG_IP_URL_BASE.format(host=hostname) + '/ltm/monitor/{type}'.format(type=monitor_type))
    except requests.exceptions.ConnectionError as e:
        return _load_connection_error(hostname, e)
    return _load_response(response)

def create_monitor(hostname, username, password, monitor_type, name, **kwargs):
    if False:
        print('Hello World!')
    "\n    A function to connect to a bigip device and create a monitor.\n\n    hostname\n        The host/address of the bigip device\n    username\n        The iControl REST username\n    password\n        The iControl REST password\n    monitor_type\n        The type of monitor to create\n    name\n        The name of the monitor to create\n    kwargs\n        Consult F5 BIGIP user guide for specific options for each monitor type.\n        Typically, tmsh arg names are used.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' bigip.create_monitor bigip admin admin http my-http-monitor timeout=10 interval=5\n    "
    bigip_session = _build_session(username, password)
    payload = {}
    payload['name'] = name
    for (key, value) in kwargs.items():
        if not key.startswith('__'):
            if key not in ['hostname', 'username', 'password', 'type']:
                key = key.replace('_', '-')
                payload[key] = value
    try:
        response = bigip_session.post(BIG_IP_URL_BASE.format(host=hostname) + '/ltm/monitor/{type}'.format(type=monitor_type), data=salt.utils.json.dumps(payload))
    except requests.exceptions.ConnectionError as e:
        return _load_connection_error(hostname, e)
    return _load_response(response)

def modify_monitor(hostname, username, password, monitor_type, name, **kwargs):
    if False:
        print('Hello World!')
    "\n    A function to connect to a bigip device and modify an existing monitor.\n\n    hostname\n        The host/address of the bigip device\n    username\n        The iControl REST username\n    password\n        The iControl REST password\n    monitor_type\n        The type of monitor to modify\n    name\n        The name of the monitor to modify\n    kwargs\n        Consult F5 BIGIP user guide for specific options for each monitor type.\n        Typically, tmsh arg names are used.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' bigip.modify_monitor bigip admin admin http my-http-monitor  timout=16 interval=6\n\n    "
    bigip_session = _build_session(username, password)
    payload = {}
    for (key, value) in kwargs.items():
        if not key.startswith('__'):
            if key not in ['hostname', 'username', 'password', 'type', 'name']:
                key = key.replace('_', '-')
                payload[key] = value
    try:
        response = bigip_session.put(BIG_IP_URL_BASE.format(host=hostname) + '/ltm/monitor/{type}/{name}'.format(type=monitor_type, name=name), data=salt.utils.json.dumps(payload))
    except requests.exceptions.ConnectionError as e:
        return _load_connection_error(hostname, e)
    return _load_response(response)

def delete_monitor(hostname, username, password, monitor_type, name):
    if False:
        i = 10
        return i + 15
    "\n    A function to connect to a bigip device and delete an existing monitor.\n\n    hostname\n        The host/address of the bigip device\n    username\n        The iControl REST username\n    password\n        The iControl REST password\n    monitor_type\n        The type of monitor to delete\n    name\n        The name of the monitor to delete\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' bigip.delete_monitor bigip admin admin http my-http-monitor\n\n    "
    bigip_session = _build_session(username, password)
    try:
        response = bigip_session.delete(BIG_IP_URL_BASE.format(host=hostname) + '/ltm/monitor/{type}/{name}'.format(type=monitor_type, name=name))
    except requests.exceptions.ConnectionError as e:
        return _load_connection_error(hostname, e)
    if _load_response(response) == '':
        return True
    else:
        return _load_response(response)

def list_profile(hostname, username, password, profile_type, name=None):
    if False:
        print('Hello World!')
    "\n    A function to connect to a bigip device and list an existing profile.  If no name is provided than all\n    profiles of the specified type will be listed.\n\n    hostname\n        The host/address of the bigip device\n    username\n        The iControl REST username\n    password\n        The iControl REST password\n    profile_type\n        The type of profile(s) to list\n    name\n        The name of the profile to list\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' bigip.list_profile bigip admin admin http my-http-profile\n\n    "
    bigip_session = _build_session(username, password)
    try:
        if name:
            response = bigip_session.get(BIG_IP_URL_BASE.format(host=hostname) + '/ltm/profile/{type}/{name}?expandSubcollections=true'.format(type=profile_type, name=name))
        else:
            response = bigip_session.get(BIG_IP_URL_BASE.format(host=hostname) + '/ltm/profile/{type}'.format(type=profile_type))
    except requests.exceptions.ConnectionError as e:
        return _load_connection_error(hostname, e)
    return _load_response(response)

def create_profile(hostname, username, password, profile_type, name, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    A function to connect to a bigip device and create a profile.\n\n    hostname\n        The host/address of the bigip device\n    username\n        The iControl REST username\n    password\n        The iControl REST password\n    profile_type\n        The type of profile to create\n    name\n        The name of the profile to create\n    kwargs\n        ``[ arg=val ] ... [arg=key1:val1,key2:val2] ...``\n\n        Consult F5 BIGIP user guide for specific options for each monitor type.\n        Typically, tmsh arg names are used.\n\n    Creating Complex Args\n        Profiles can get pretty complicated in terms of the amount of possible\n        config options. Use the following shorthand to create complex arguments such\n        as lists, dictionaries, and lists of dictionaries. An option is also\n        provided to pass raw json as well.\n\n        lists ``[i,i,i]``:\n            ``param=\'item1,item2,item3\'``\n\n        Dictionary ``[k:v,k:v,k,v]``:\n            ``param=\'key-1:val-1,key-2:val2,key-3:va-3\'``\n\n        List of Dictionaries ``[k:v,k:v|k:v,k:v|k:v,k:v]``:\n           ``param=\'key-1:val-1,key-2:val-2|key-1:val-1,key-2:val-2|key-1:val-1,key-2:val-2\'``\n\n        JSON: ``\'j{ ... }j\'``:\n           ``cert-key-chain=\'j{ "default": { "cert": "default.crt", "chain": "default.crt", "key": "default.key" } }j\'``\n\n        Escaping Delimiters:\n            Use ``\\,`` or ``\\:`` or ``\\|`` to escape characters which shouldn\'t\n            be treated as delimiters i.e. ``ciphers=\'DEFAULT\\:!SSLv3\'``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' bigip.create_profile bigip admin admin http my-http-profile defaultsFrom=\'/Common/http\'\n        salt \'*\' bigip.create_profile bigip admin admin http my-http-profile defaultsFrom=\'/Common/http\' \\\n            enforcement=maxHeaderCount:3200,maxRequests:10\n\n    '
    bigip_session = _build_session(username, password)
    payload = {}
    payload['name'] = name
    for (key, value) in kwargs.items():
        if not key.startswith('__'):
            if key not in ['hostname', 'username', 'password', 'profile_type']:
                key = key.replace('_', '-')
                try:
                    payload[key] = _set_value(value)
                except salt.exceptions.CommandExecutionError:
                    return 'Error: Unable to Parse JSON data for parameter: {key}\n{value}'.format(key=key, value=value)
    try:
        response = bigip_session.post(BIG_IP_URL_BASE.format(host=hostname) + '/ltm/profile/{type}'.format(type=profile_type), data=salt.utils.json.dumps(payload))
    except requests.exceptions.ConnectionError as e:
        return _load_connection_error(hostname, e)
    return _load_response(response)

def modify_profile(hostname, username, password, profile_type, name, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    A function to connect to a bigip device and create a profile.\n\n    A function to connect to a bigip device and create a profile.\n\n    hostname\n        The host/address of the bigip device\n    username\n        The iControl REST username\n    password\n        The iControl REST password\n    profile_type\n        The type of profile to create\n    name\n        The name of the profile to create\n    kwargs\n        ``[ arg=val ] ... [arg=key1:val1,key2:val2] ...``\n\n        Consult F5 BIGIP user guide for specific options for each monitor type.\n        Typically, tmsh arg names are used.\n\n    Creating Complex Args\n\n        Profiles can get pretty complicated in terms of the amount of possible\n        config options. Use the following shorthand to create complex arguments such\n        as lists, dictionaries, and lists of dictionaries. An option is also\n        provided to pass raw json as well.\n\n        lists ``[i,i,i]``:\n            ``param=\'item1,item2,item3\'``\n\n        Dictionary ``[k:v,k:v,k,v]``:\n            ``param=\'key-1:val-1,key-2:val2,key-3:va-3\'``\n\n        List of Dictionaries ``[k:v,k:v|k:v,k:v|k:v,k:v]``:\n           ``param=\'key-1:val-1,key-2:val-2|key-1:val-1,key-2:val-2|key-1:val-1,key-2:val-2\'``\n\n        JSON: ``\'j{ ... }j\'``:\n           ``cert-key-chain=\'j{ "default": { "cert": "default.crt", "chain": "default.crt", "key": "default.key" } }j\'``\n\n        Escaping Delimiters:\n            Use ``\\,`` or ``\\:`` or ``\\|`` to escape characters which shouldn\'t\n            be treated as delimiters i.e. ``ciphers=\'DEFAULT\\:!SSLv3\'``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' bigip.modify_profile bigip admin admin http my-http-profile defaultsFrom=\'/Common/http\'\n\n        salt \'*\' bigip.modify_profile bigip admin admin http my-http-profile defaultsFrom=\'/Common/http\' \\\n            enforcement=maxHeaderCount:3200,maxRequests:10\n\n        salt \'*\' bigip.modify_profile bigip admin admin client-ssl my-client-ssl-1 retainCertificate=false \\\n            ciphers=\'DEFAULT\\:!SSLv3\'\n            cert_key_chain=\'j{ "default": { "cert": "default.crt", "chain": "default.crt", "key": "default.key" } }j\'\n    '
    bigip_session = _build_session(username, password)
    payload = {}
    payload['name'] = name
    for (key, value) in kwargs.items():
        if not key.startswith('__'):
            if key not in ['hostname', 'username', 'password', 'profile_type']:
                key = key.replace('_', '-')
                try:
                    payload[key] = _set_value(value)
                except salt.exceptions.CommandExecutionError:
                    return 'Error: Unable to Parse JSON data for parameter: {key}\n{value}'.format(key=key, value=value)
    try:
        response = bigip_session.put(BIG_IP_URL_BASE.format(host=hostname) + '/ltm/profile/{type}/{name}'.format(type=profile_type, name=name), data=salt.utils.json.dumps(payload))
    except requests.exceptions.ConnectionError as e:
        return _load_connection_error(hostname, e)
    return _load_response(response)

def delete_profile(hostname, username, password, profile_type, name):
    if False:
        print('Hello World!')
    "\n    A function to connect to a bigip device and delete an existing profile.\n\n    hostname\n        The host/address of the bigip device\n    username\n        The iControl REST username\n    password\n        The iControl REST password\n    profile_type\n        The type of profile to delete\n    name\n        The name of the profile to delete\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' bigip.delete_profile bigip admin admin http my-http-profile\n\n    "
    bigip_session = _build_session(username, password)
    try:
        response = bigip_session.delete(BIG_IP_URL_BASE.format(host=hostname) + '/ltm/profile/{type}/{name}'.format(type=profile_type, name=name))
    except requests.exceptions.ConnectionError as e:
        return _load_connection_error(hostname, e)
    if _load_response(response) == '':
        return True
    else:
        return _load_response(response)