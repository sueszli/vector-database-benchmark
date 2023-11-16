"""
Manage HP ILO

:depends: hponcfg (SmartStart Scripting Toolkit Linux Edition)
"""
import logging
import os
import tempfile
import xml.etree.ElementTree as ET
import salt.utils.path
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Make sure hponcfg tool is accessible\n    '
    if salt.utils.path.which('hponcfg'):
        return True
    return (False, 'ilo execution module not loaded: the hponcfg binary is not in the path.')

def __execute_cmd(name, xml):
    if False:
        i = 10
        return i + 15
    '\n    Execute ilom commands\n    '
    ret = {name.replace('_', ' '): {}}
    id_num = 0
    tmp_dir = os.path.join(__opts__['cachedir'], 'tmp')
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
    with tempfile.NamedTemporaryFile(dir=tmp_dir, prefix=name + str(os.getpid()), suffix='.xml', mode='w', delete=False) as fh:
        tmpfilename = fh.name
        fh.write(xml)
    cmd = __salt__['cmd.run_all']('hponcfg -f {}'.format(tmpfilename))
    __salt__['file.remove'](tmpfilename)
    if cmd['retcode'] != 0:
        for i in cmd['stderr'].splitlines():
            if i.startswith('     MESSAGE='):
                return {'Failed': i.split('=')[-1]}
        return False
    try:
        for i in ET.fromstring(''.join(cmd['stdout'].splitlines()[3:-1])):
            if ret[name.replace('_', ' ')].get(i.tag, False):
                ret[name.replace('_', ' ')].update({i.tag + '_' + str(id_num): i.attrib})
                id_num += 1
            else:
                ret[name.replace('_', ' ')].update({i.tag: i.attrib})
    except SyntaxError:
        return True
    return ret

def global_settings():
    if False:
        return 10
    "\n    Show global settings\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ilo.global_settings\n    "
    _xml = '<!-- Sample file for Get Global command -->\n              <RIBCL VERSION="2.0">\n                 <LOGIN USER_LOGIN="x" PASSWORD="x">\n                   <RIB_INFO MODE="read">\n                     <GET_GLOBAL_SETTINGS />\n                   </RIB_INFO>\n                 </LOGIN>\n               </RIBCL>'
    return __execute_cmd('Global_Settings', _xml)

def set_http_port(port=80):
    if False:
        i = 10
        return i + 15
    "\n    Configure the port HTTP should listen on\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ilo.set_http_port 8080\n    "
    _current = global_settings()
    if _current['Global Settings']['HTTP_PORT']['VALUE'] == port:
        return True
    _xml = '<RIBCL VERSION="2.0">\n                <LOGIN USER_LOGIN="adminname" PASSWORD="password">\n                  <RIB_INFO MODE="write">\n                    <MOD_GLOBAL_SETTINGS>\n                      <HTTP_PORT value="{}"/>\n                    </MOD_GLOBAL_SETTINGS>\n                  </RIB_INFO>\n                </LOGIN>\n              </RIBCL>'.format(port)
    return __execute_cmd('Set_HTTP_Port', _xml)

def set_https_port(port=443):
    if False:
        return 10
    "\n    Configure the port HTTPS should listen on\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ilo.set_https_port 4334\n    "
    _current = global_settings()
    if _current['Global Settings']['HTTP_PORT']['VALUE'] == port:
        return True
    _xml = '<RIBCL VERSION="2.0">\n                <LOGIN USER_LOGIN="adminname" PASSWORD="password">\n                  <RIB_INFO MODE="write">\n                    <MOD_GLOBAL_SETTINGS>\n                      <HTTPS_PORT value="{}"/>\n                    </MOD_GLOBAL_SETTINGS>\n                  </RIB_INFO>\n                </LOGIN>\n              </RIBCL>'.format(port)
    return __execute_cmd('Set_HTTPS_Port', _xml)

def enable_ssh():
    if False:
        while True:
            i = 10
    "\n    Enable the SSH daemon\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ilo.enable_ssh\n    "
    _current = global_settings()
    if _current['Global Settings']['SSH_STATUS']['VALUE'] == 'Y':
        return True
    _xml = '<RIBCL VERSION="2.0">\n                <LOGIN USER_LOGIN="adminname" PASSWORD="password">\n                  <RIB_INFO MODE="write">\n                    <MOD_GLOBAL_SETTINGS>\n                      <SSH_STATUS value="Yes"/>\n                    </MOD_GLOBAL_SETTINGS>\n                  </RIB_INFO>\n                </LOGIN>\n              </RIBCL>'
    return __execute_cmd('Enable_SSH', _xml)

def disable_ssh():
    if False:
        i = 10
        return i + 15
    "\n    Disable the SSH daemon\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ilo.disable_ssh\n    "
    _current = global_settings()
    if _current['Global Settings']['SSH_STATUS']['VALUE'] == 'N':
        return True
    _xml = '<RIBCL VERSION="2.0">\n                <LOGIN USER_LOGIN="adminname" PASSWORD="password">\n                  <RIB_INFO MODE="write">\n                    <MOD_GLOBAL_SETTINGS>\n                      <SSH_STATUS value="No"/>\n                    </MOD_GLOBAL_SETTINGS>\n                  </RIB_INFO>\n                </LOGIN>\n              </RIBCL>'
    return __execute_cmd('Disable_SSH', _xml)

def set_ssh_port(port=22):
    if False:
        return 10
    "\n    Enable SSH on a user defined port\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ilo.set_ssh_port 2222\n    "
    _current = global_settings()
    if _current['Global Settings']['SSH_PORT']['VALUE'] == port:
        return True
    _xml = '<RIBCL VERSION="2.0">\n                <LOGIN USER_LOGIN="adminname" PASSWORD="password">\n                  <RIB_INFO MODE="write">\n                    <MOD_GLOBAL_SETTINGS>\n                       <SSH_PORT value="{}"/>\n                    </MOD_GLOBAL_SETTINGS>\n                  </RIB_INFO>\n                </LOGIN>\n              </RIBCL>'.format(port)
    return __execute_cmd('Configure_SSH_Port', _xml)

def set_ssh_key(public_key):
    if False:
        print('Hello World!')
    '\n    Configure SSH public keys for specific users\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' ilo.set_ssh_key "ssh-dss AAAAB3NzaC1kc3MAAACBA... damian"\n\n    The SSH public key needs to be DSA and the last argument in the key needs\n    to be the username (case-senstive) of the ILO username.\n    '
    _xml = '<RIBCL VERSION="2.0">\n                <LOGIN USER_LOGIN="adminname" PASSWORD="password">\n                  <RIB_INFO MODE="write">\n                    <IMPORT_SSH_KEY>\n                      -----BEGIN SSH KEY-----\n                      {}\n                      -----END SSH KEY-----\n                    </IMPORT_SSH_KEY>\n                  </RIB_INFO>\n                </LOGIN>\n              </RIBCL>'.format(public_key)
    return __execute_cmd('Import_SSH_Publickey', _xml)

def delete_ssh_key(username):
    if False:
        return 10
    "\n    Delete a users SSH key from the ILO\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ilo.delete_user_sshkey damian\n    "
    _xml = '<RIBCL VERSION="2.0">\n                <LOGIN USER_LOGIN="admin" PASSWORD="admin123">\n                  <USER_INFO MODE="write">\n                    <MOD_USER USER_LOGIN="{}">\n                      <DEL_USERS_SSH_KEY/>\n                    </MOD_USER>\n                  </USER_INFO>\n                </LOGIN>\n              </RIBCL>'.format(username)
    return __execute_cmd('Delete_user_SSH_key', _xml)

def list_users():
    if False:
        i = 10
        return i + 15
    "\n    List all users\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ilo.list_users\n    "
    _xml = '<RIBCL VERSION="2.0">\n                <LOGIN USER_LOGIN="x" PASSWORD="x">\n                    <USER_INFO MODE="read">\n                      <GET_ALL_USERS />\n                    </USER_INFO>\n                </LOGIN>\n              </RIBCL>'
    return __execute_cmd('All_users', _xml)

def list_users_info():
    if False:
        i = 10
        return i + 15
    "\n    List all users in detail\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ilo.list_users_info\n    "
    _xml = '<RIBCL VERSION="2.0">\n                <LOGIN USER_LOGIN="adminname" PASSWORD="password">\n                  <USER_INFO MODE="read">\n                    <GET_ALL_USER_INFO />\n                  </USER_INFO>\n                </LOGIN>\n              </RIBCL>'
    return __execute_cmd('All_users_info', _xml)

def create_user(name, password, *privileges):
    if False:
        i = 10
        return i + 15
    "\n    Create user\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ilo.create_user damian secretagent VIRTUAL_MEDIA_PRIV\n\n    If no permissions are specify the user will only have a read-only account.\n\n    Supported privelges:\n\n    * ADMIN_PRIV\n      Enables the user to administer user accounts.\n\n    * REMOTE_CONS_PRIV\n      Enables the user to access the Remote Console functionality.\n\n    * RESET_SERVER_PRIV\n      Enables the user to remotely manipulate the server power setting.\n\n    * VIRTUAL_MEDIA_PRIV\n      Enables the user permission to access the virtual media functionality.\n\n    * CONFIG_ILO_PRIV\n      Enables the user to configure iLO settings.\n    "
    _priv = ['ADMIN_PRIV', 'REMOTE_CONS_PRIV', 'RESET_SERVER_PRIV', 'VIRTUAL_MEDIA_PRIV', 'CONFIG_ILO_PRIV']
    _xml = '<RIBCL version="2.2">\n                <LOGIN USER_LOGIN="x" PASSWORD="y">\n                  <RIB_INFO mode="write">\n                    <MOD_GLOBAL_SETTINGS>\n                      <MIN_PASSWORD VALUE="7"/>\n                    </MOD_GLOBAL_SETTINGS>\n                  </RIB_INFO>\n\n                 <USER_INFO MODE="write">\n                   <ADD_USER USER_NAME="{0}" USER_LOGIN="{0}" PASSWORD="{1}">\n                     {2}\n                   </ADD_USER>\n                 </USER_INFO>\n               </LOGIN>\n             </RIBCL>'.format(name, password, '\n'.join(['<{} value="Y" />'.format(i.upper()) for i in privileges if i.upper() in _priv]))
    return __execute_cmd('Create_user', _xml)

def delete_user(username):
    if False:
        i = 10
        return i + 15
    "\n    Delete a user\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ilo.delete_user damian\n    "
    _xml = '<RIBCL VERSION="2.0">\n                <LOGIN USER_LOGIN="adminname" PASSWORD="password">\n                  <USER_INFO MODE="write">\n                    <DELETE_USER USER_LOGIN="{}" />\n                  </USER_INFO>\n                </LOGIN>\n              </RIBCL>'.format(username)
    return __execute_cmd('Delete_user', _xml)

def get_user(username):
    if False:
        while True:
            i = 10
    "\n    Returns local user information, excluding the password\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ilo.get_user damian\n    "
    _xml = '<RIBCL VERSION="2.0">\n                <LOGIN USER_LOGIN="adminname" PASSWORD="password">\n                  <USER_INFO MODE="read">\n                    <GET_USER USER_LOGIN="{}" />\n                  </USER_INFO>\n                </LOGIN>\n              </RIBCL>'.format(username)
    return __execute_cmd('User_Info', _xml)

def change_username(old_username, new_username):
    if False:
        print('Hello World!')
    "\n    Change a username\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ilo.change_username damian diana\n    "
    _xml = '<RIBCL VERSION="2.0">\n                <LOGIN USER_LOGIN="adminname" PASSWORD="password">\n                  <USER_INFO MODE="write">\n                    <MOD_USER USER_LOGIN="{0}">\n                      <USER_NAME value="{1}"/>\n                      <USER_LOGIN value="{1}"/>\n                    </MOD_USER>\n                  </USER_INFO>\n                </LOGIN>\n              </RIBCL>'.format(old_username, new_username)
    return __execute_cmd('Change_username', _xml)

def change_password(username, password):
    if False:
        return 10
    "\n    Reset a users password\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ilo.change_password damianMyerscough\n    "
    _xml = '<RIBCL VERSION="2.0">\n                <LOGIN USER_LOGIN="adminname" PASSWORD="password">\n                  <USER_INFO MODE="write">\n                    <MOD_USER USER_LOGIN="{}">\n                      <PASSWORD value="{}"/>\n                    </MOD_USER>\n                  </USER_INFO>\n                </LOGIN>\n              </RIBCL>'.format(username, password)
    return __execute_cmd('Change_password', _xml)

def network():
    if False:
        return 10
    "\n    Grab the current network settings\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ilo.network\n    "
    _xml = '<RIBCL VERSION="2.0">\n                <LOGIN USER_LOGIN="adminname" PASSWORD="password">\n                  <RIB_INFO MODE="read">\n                    <GET_NETWORK_SETTINGS/>\n                  </RIB_INFO>\n                </LOGIN>\n              </RIBCL>'
    return __execute_cmd('Network_Settings', _xml)

def configure_network(ip, netmask, gateway):
    if False:
        i = 10
        return i + 15
    "\n    Configure Network Interface\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ilo.configure_network [IP ADDRESS] [NETMASK] [GATEWAY]\n    "
    current = network()
    if ip in current['Network Settings']['IP_ADDRESS']['VALUE'] and netmask in current['Network Settings']['SUBNET_MASK']['VALUE'] and (gateway in current['Network Settings']['GATEWAY_IP_ADDRESS']['VALUE']):
        return True
    _xml = '<RIBCL VERSION="2.0">\n                <LOGIN USER_LOGIN="adminname" PASSWORD="password">\n                  <RIB_INFO MODE="write">\n                    <MOD_NETWORK_SETTINGS>\n                      <IP_ADDRESS value="{}"/>\n                      <SUBNET_MASK value="{}"/>\n                      <GATEWAY_IP_ADDRESS value="{}"/>\n                    </MOD_NETWORK_SETTINGS>\n                  </RIB_INFO>\n                </LOGIN>\n              </RIBCL> '.format(ip, netmask, gateway)
    return __execute_cmd('Configure_Network', _xml)

def enable_dhcp():
    if False:
        print('Hello World!')
    "\n    Enable DHCP\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ilo.enable_dhcp\n    "
    current = network()
    if current['Network Settings']['DHCP_ENABLE']['VALUE'] == 'Y':
        return True
    _xml = '<RIBCL VERSION="2.0">\n                <LOGIN USER_LOGIN="adminname" PASSWORD="password">\n                  <RIB_INFO MODE="write">\n                    <MOD_NETWORK_SETTINGS>\n                      <DHCP_ENABLE value="Yes"/>\n                    </MOD_NETWORK_SETTINGS>\n                  </RIB_INFO>\n                </LOGIN>\n              </RIBCL>'
    return __execute_cmd('Enable_DHCP', _xml)

def disable_dhcp():
    if False:
        i = 10
        return i + 15
    "\n    Disable DHCP\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ilo.disable_dhcp\n    "
    current = network()
    if current['Network Settings']['DHCP_ENABLE']['VALUE'] == 'N':
        return True
    _xml = '<RIBCL VERSION="2.0">\n                <LOGIN USER_LOGIN="adminname" PASSWORD="password">\n                  <RIB_INFO MODE="write">\n                    <MOD_NETWORK_SETTINGS>\n                      <DHCP_ENABLE value="No"/>\n                    </MOD_NETWORK_SETTINGS>\n                  </RIB_INFO>\n                </LOGIN>\n              </RIBCL>'
    return __execute_cmd('Disable_DHCP', _xml)

def configure_snmp(community, snmp_port=161, snmp_trapport=161):
    if False:
        return 10
    "\n    Configure SNMP\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ilo.configure_snmp [COMMUNITY STRING] [SNMP PORT] [SNMP TRAP PORT]\n    "
    _xml = '<RIBCL VERSION="2.2">\n                <LOGIN USER_LOGIN="x" PASSWORD="y">\n                  <RIB_INFO mode="write">\n                    <MOD_GLOBAL_SETTINGS>\n                      <SNMP_ACCESS_ENABLED VALUE="Yes"/>\n                      <SNMP_PORT VALUE="{}"/>\n                      <SNMP_TRAP_PORT VALUE="{}"/>\n                    </MOD_GLOBAL_SETTINGS>\n\n                   <MOD_SNMP_IM_SETTINGS>\n                     <SNMP_ADDRESS_1 VALUE=""/>\n                     <SNMP_ADDRESS_1_ROCOMMUNITY VALUE="{}"/>\n                     <SNMP_ADDRESS_1_TRAPCOMMUNITY VERSION="" VALUE=""/>\n                     <RIB_TRAPS VALUE="Y"/>\n                     <OS_TRAPS VALUE="Y"/>\n                     <SNMP_PASSTHROUGH_STATUS VALUE="N"/>\n                  </MOD_SNMP_IM_SETTINGS>\n                </RIB_INFO>\n              </LOGIN>\n           </RIBCL>'.format(snmp_port, snmp_trapport, community)
    return __execute_cmd('Configure_SNMP', _xml)