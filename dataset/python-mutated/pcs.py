"""
Configure a Pacemaker/Corosync cluster with PCS
===============================================

Configure Pacemaker/Cororsync clusters with the
Pacemaker/Cororsync conifguration system (PCS)

:depends: pcs

.. versionadded:: 2016.3.0
"""
import logging
import salt.utils.path
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        return 10
    '\n    Only load if pcs package is installed\n    '
    if salt.utils.path.which('pcs'):
        return 'pcs'
    return (False, 'Missing dependency: pcs')

def __use_new_commands():
    if False:
        for i in range(10):
            print('nop')
    '\n    The command line arguments of pcs changed after version 0.10\n    This will return True if the new arguments are needed and\n    false if the old ones are needed\n    '
    pcs_version = __salt__['pkg.version']('pcs')
    log.debug('PCS package version %s', pcs_version)
    if __salt__['pkg.version_cmp'](pcs_version, '0.10') == 1:
        log.debug('New version, new command')
        return True
    else:
        log.debug('Old Version')
        return False

def item_show(item, item_id=None, item_type=None, show='show', extra_args=None, cibfile=None):
    if False:
        print('Hello World!')
    '\n    Show an item via pcs command\n    (mainly for use with the pcs state module)\n\n    item\n        config, property, resource, constraint etc.\n    item_id\n        id of the item\n    item_type\n        item type\n    show\n        show command (probably None, default: show or status for newer implementation)\n    extra_args\n        additional options for the pcs command\n    cibfile\n        use cibfile instead of the live CIB\n    '
    new_commands = __use_new_commands()
    cmd = ['pcs']
    if isinstance(cibfile, str):
        cmd += ['-f', cibfile]
    if isinstance(item, str):
        cmd += [item]
    elif isinstance(item, (list, tuple)):
        cmd += item
    if item in ['constraint']:
        cmd += [item_type]
    if new_commands and (item != 'config' and item != 'constraint' and (item != 'property')):
        if show == 'show':
            show = 'config'
        elif isinstance(show, (list, tuple)):
            for (index, value) in enumerate(show):
                if show[index] == 'show':
                    show[index] = 'config'
    if isinstance(show, str):
        cmd += [show]
    elif isinstance(show, (list, tuple)):
        cmd += show
    if isinstance(item_id, str):
        cmd += [item_id]
    if isinstance(extra_args, (list, tuple)):
        cmd += extra_args
    if item in ['constraint']:
        if not isinstance(extra_args, (list, tuple)) or '--full' not in extra_args:
            cmd += ['--full']
    log.debug('Running item show %s', cmd)
    return __salt__['cmd.run_all'](cmd, output_loglevel='trace', python_shell=False)

def item_create(item, item_id, item_type, create='create', extra_args=None, cibfile=None):
    if False:
        i = 10
        return i + 15
    '\n    Create an item via pcs command\n    (mainly for use with the pcs state module)\n\n    item\n        config, property, resource, constraint etc.\n    item_id\n        id of the item\n    item_type\n        item type\n    create\n        create command (create or set f.e., default: create)\n    extra_args\n        additional options for the pcs command\n    cibfile\n        use cibfile instead of the live CIB\n    '
    cmd = ['pcs']
    if isinstance(cibfile, str):
        cmd += ['-f', cibfile]
    if isinstance(item, str):
        cmd += [item]
    elif isinstance(item, (list, tuple)):
        cmd += item
    if item in ['constraint']:
        if isinstance(item_type, str):
            cmd += [item_type]
    if isinstance(create, str):
        cmd += [create]
    elif isinstance(create, (list, tuple)):
        cmd += create
    if item not in ['constraint']:
        cmd += [item_id]
        if isinstance(item_type, str):
            cmd += [item_type]
    if isinstance(extra_args, (list, tuple)):
        if item in ['constraint']:
            extra_args = extra_args + ['id={}'.format(item_id)]
        cmd += extra_args
    return __salt__['cmd.run_all'](cmd, output_loglevel='trace', python_shell=False)

def auth(nodes, pcsuser='hacluster', pcspasswd='hacluster', extra_args=None):
    if False:
        return 10
    "\n    Authorize nodes to the cluster\n\n    nodes\n        a list of nodes which should be authorized to the cluster\n    pcsuser\n        user for communitcation with PCS (default: hacluster)\n    pcspasswd\n        password for pcsuser (default: hacluster)\n    extra_args\n        list of extra option for the 'pcs cluster auth' command. The newer cluster host command has no extra args and so will ignore it.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pcs.auth nodes='[ node1.example.org, node2.example.org ]' pcsuser=hacluster pcspasswd=hoonetorg extra_args=[ '--force' ]\n    "
    if __use_new_commands():
        cmd = ['pcs', 'host', 'auth']
    else:
        cmd = ['pcs', 'cluster', 'auth']
    cmd.extend(['-u', pcsuser, '-p', pcspasswd])
    if not __use_new_commands() and isinstance(extra_args, (list, tuple)):
        cmd += extra_args
    cmd += nodes
    return __salt__['cmd.run_all'](cmd, output_loglevel='trace', python_shell=False)

def is_auth(nodes, pcsuser='hacluster', pcspasswd='hacluster'):
    if False:
        i = 10
        return i + 15
    "\n    Check if nodes are already authorized\n\n    nodes\n        a list of nodes to be checked for authorization to the cluster\n    pcsuser\n        user for communitcation with PCS (default: hacluster)\n    pcspasswd\n        password for pcsuser (default: hacluster)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pcs.is_auth nodes='[node1.example.org, node2.example.org]' pcsuser=hacluster pcspasswd=hoonetorg\n    "
    if __use_new_commands():
        cmd = ['pcs', 'host', 'auth', '-u', pcsuser, '-p', pcspasswd]
    else:
        cmd = ['pcs', 'cluster', 'auth']
    cmd += nodes
    return __salt__['cmd.run_all'](cmd, stdin='\n\n', output_loglevel='trace', python_shell=False)

def cluster_setup(nodes, pcsclustername='pcscluster', extra_args=None):
    if False:
        while True:
            i = 10
    "\n    Setup pacemaker cluster via pcs command\n\n    nodes\n        a list of nodes which should be set up\n    pcsclustername\n        Name of the Pacemaker cluster (default: pcscluster)\n    extra_args\n        list of extra option for the 'pcs cluster setup' command\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pcs.cluster_setup nodes='[ node1.example.org, node2.example.org ]' pcsclustername=pcscluster\n    "
    cmd = ['pcs', 'cluster', 'setup']
    if __use_new_commands():
        cmd += [pcsclustername]
    else:
        cmd += ['--name', pcsclustername]
    cmd += nodes
    if isinstance(extra_args, (list, tuple)):
        cmd += extra_args
    log.debug('Running cluster setup: %s', cmd)
    return __salt__['cmd.run_all'](cmd, output_loglevel='trace', python_shell=False)

def cluster_destroy(extra_args=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Destroy corosync cluster using the pcs command\n\n    extra_args\n        list of extra option for the 'pcs cluster destroy' command (only really --all)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pcs.cluster_destroy extra_args=--all\n    "
    cmd = ['pcs', 'cluster', 'destroy']
    if isinstance(extra_args, (list, tuple)):
        cmd += extra_args
    log.debug('Running cluster destroy: %s', cmd)
    return __salt__['cmd.run_all'](cmd, output_loglevel='trace', python_shell=False)

def cluster_node_add(node, extra_args=None):
    if False:
        print('Hello World!')
    "\n    Add a node to the pacemaker cluster via pcs command\n\n    node\n        node that should be added\n    extra_args\n        list of extra option for the 'pcs cluster node add' command\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pcs.cluster_node_add node=node2.example.org\n    "
    cmd = ['pcs', 'cluster', 'node', 'add']
    cmd += [node]
    if isinstance(extra_args, (list, tuple)):
        cmd += extra_args
    return __salt__['cmd.run_all'](cmd, output_loglevel='trace', python_shell=False)

def cib_create(cibfile, scope='configuration', extra_args=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Create a CIB-file from the current CIB of the cluster\n\n    cibfile\n        name/path of the file containing the CIB\n    scope\n        specific section of the CIB (default: configuration)\n    extra_args\n        additional options for creating the CIB-file\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pcs.cib_create cibfile='/tmp/VIP_apache_1.cib' scope=False\n    "
    cmd = ['pcs', 'cluster', 'cib', cibfile]
    if isinstance(scope, str):
        cmd += ['scope={}'.format(scope)]
    if isinstance(extra_args, (list, tuple)):
        cmd += extra_args
    return __salt__['cmd.run_all'](cmd, output_loglevel='trace', python_shell=False)

def cib_push(cibfile, scope='configuration', extra_args=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Push a CIB-file as the new CIB to the cluster\n\n    cibfile\n        name/path of the file containing the CIB\n    scope\n        specific section of the CIB (default: configuration)\n    extra_args\n        additional options for creating the CIB-file\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pcs.cib_push cibfile='/tmp/VIP_apache_1.cib' scope=False\n    "
    cmd = ['pcs', 'cluster', 'cib-push', cibfile]
    if isinstance(scope, str):
        cmd += ['scope={}'.format(scope)]
    if isinstance(extra_args, (list, tuple)):
        cmd += extra_args
    return __salt__['cmd.run_all'](cmd, output_loglevel='trace', python_shell=False)

def config_show(cibfile=None):
    if False:
        while True:
            i = 10
    "\n    Show config of cluster\n\n    cibfile\n        name/path of the file containing the CIB\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pcs.config_show cibfile='/tmp/cib_for_galera'\n    "
    return item_show(item='config', item_id=None, extra_args=None, cibfile=cibfile)

def prop_show(prop, extra_args=None, cibfile=None):
    if False:
        print('Hello World!')
    "\n    Show the value of a cluster property\n\n    prop\n        name of the property\n    extra_args\n        additional options for the pcs property command\n    cibfile\n        use cibfile instead of the live CIB\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pcs.prop_show cibfile='/tmp/2_node_cluster.cib' prop='no-quorum-policy' cibfile='/tmp/2_node_cluster.cib'\n    "
    return item_show(item='property', item_id=prop, extra_args=extra_args, cibfile=cibfile)

def prop_set(prop, value, extra_args=None, cibfile=None):
    if False:
        i = 10
        return i + 15
    "\n    Set the value of a cluster property\n\n    prop\n        name of the property\n    value\n        value of the property prop\n    extra_args\n        additional options for the pcs property command\n    cibfile\n        use cibfile instead of the live CIB\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pcs.prop_set prop='no-quorum-policy' value='ignore' cibfile='/tmp/2_node_cluster.cib'\n    "
    return item_create(item='property', item_id='{}={}'.format(prop, value), item_type=None, create='set', extra_args=extra_args, cibfile=cibfile)

def stonith_show(stonith_id, extra_args=None, cibfile=None):
    if False:
        while True:
            i = 10
    "\n    Show the value of a cluster stonith\n\n    stonith_id\n        name for the stonith resource\n    extra_args\n        additional options for the pcs stonith command\n    cibfile\n        use cibfile instead of the live CIB\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pcs.stonith_show stonith_id='eps_fence' cibfile='/tmp/2_node_cluster.cib'\n    "
    return item_show(item='stonith', item_id=stonith_id, extra_args=extra_args, cibfile=cibfile)

def stonith_create(stonith_id, stonith_device_type, stonith_device_options=None, cibfile=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a stonith resource via pcs command\n\n    stonith_id\n        name for the stonith resource\n    stonith_device_type\n        name of the stonith agent fence_eps, fence_xvm f.e.\n    stonith_device_options\n        additional options for creating the stonith resource\n    cibfile\n        use cibfile instead of the live CIB for manipulation\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pcs.stonith_create stonith_id=\'eps_fence\' stonith_device_type=\'fence_eps\'\n                                    stonith_device_options="[\'pcmk_host_map=node1.example.org:01;node2.example.org:02\', \'ipaddr=myepsdevice.example.org\', \'action=reboot\', \'power_wait=5\', \'verbose=1\', \'debug=/var/log/pcsd/eps_fence.log\', \'login=hidden\', \'passwd=hoonetorg\']" cibfile=\'/tmp/cib_for_stonith.cib\'\n    '
    return item_create(item='stonith', item_id=stonith_id, item_type=stonith_device_type, extra_args=stonith_device_options, cibfile=cibfile)

def resource_show(resource_id, extra_args=None, cibfile=None):
    if False:
        return 10
    "\n    Show a resource via pcs command\n\n    resource_id\n        name of the resource\n    extra_args\n        additional options for the pcs command\n    cibfile\n        use cibfile instead of the live CIB\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pcs.resource_show resource_id='galera' cibfile='/tmp/cib_for_galera.cib'\n    "
    return item_show(item='resource', item_id=resource_id, extra_args=extra_args, cibfile=cibfile)

def resource_create(resource_id, resource_type, resource_options=None, cibfile=None):
    if False:
        return 10
    '\n    Create a resource via pcs command\n\n    resource_id\n        name for the resource\n    resource_type\n        resource type (f.e. ocf:heartbeat:IPaddr2 or VirtualIP)\n    resource_options\n        additional options for creating the resource\n    cibfile\n        use cibfile instead of the live CIB for manipulation\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pcs.resource_create resource_id=\'galera\' resource_type=\'ocf:heartbeat:galera\' resource_options="[\'wsrep_cluster_address=gcomm://node1.example.org,node2.example.org,node3.example.org\', \'--master\']" cibfile=\'/tmp/cib_for_galera.cib\'\n    '
    return item_create(item='resource', item_id=resource_id, item_type=resource_type, extra_args=resource_options, cibfile=cibfile)