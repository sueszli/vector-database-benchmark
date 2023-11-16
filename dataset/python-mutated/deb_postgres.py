"""
Module to provide Postgres compatibility to salt for debian family specific tools.

"""
import logging
import shlex
import salt.utils.path
log = logging.getLogger(__name__)
__virtualname__ = 'postgres'

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Only load this module if the pg_createcluster bin exists\n    '
    if salt.utils.path.which('pg_createcluster'):
        return __virtualname__
    return (False, 'postgres execution module not loaded: pg_createcluste command not found.')

def cluster_create(version, name='main', port=None, locale=None, encoding=None, datadir=None, allow_group_access=None, data_checksums=None, wal_segsize=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Adds a cluster to the Postgres server.\n\n    .. warning:\n\n       Only works for debian family distros so far.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' postgres.cluster_create '9.3'\n\n        salt '*' postgres.cluster_create '9.3' 'main'\n\n        salt '*' postgres.cluster_create '9.3' locale='fr_FR'\n\n        salt '*' postgres.cluster_create '11' data_checksums=True wal_segsize='32'\n    "
    cmd = [salt.utils.path.which('pg_createcluster')]
    if port:
        cmd += ['--port', str(port)]
    if locale:
        cmd += ['--locale', locale]
    if encoding:
        cmd += ['--encoding', encoding]
    if datadir:
        cmd += ['--datadir', datadir]
    cmd += [str(version), name]
    if allow_group_access or data_checksums or wal_segsize:
        cmd += ['--']
    if allow_group_access is True:
        cmd += ['--allow-group-access']
    if data_checksums is True:
        cmd += ['--data-checksums']
    if wal_segsize:
        cmd += ['--wal-segsize', wal_segsize]
    cmdstr = ' '.join([shlex.quote(c) for c in cmd])
    ret = __salt__['cmd.run_all'](cmdstr, python_shell=False)
    if ret.get('retcode', 0) != 0:
        log.error('Error creating a Postgresql cluster %s/%s', version, name)
        return False
    return ret

def cluster_list(verbose=False):
    if False:
        print('Hello World!')
    "\n    Return a list of cluster of Postgres server (tuples of version and name).\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' postgres.cluster_list\n\n        salt '*' postgres.cluster_list verbose=True\n    "
    cmd = [salt.utils.path.which('pg_lsclusters'), '--no-header']
    ret = __salt__['cmd.run_all'](' '.join([shlex.quote(c) for c in cmd]))
    if ret.get('retcode', 0) != 0:
        log.error('Error listing clusters')
    cluster_dict = _parse_pg_lscluster(ret['stdout'])
    if verbose:
        return cluster_dict
    return cluster_dict.keys()

def cluster_exists(version, name='main'):
    if False:
        print('Hello World!')
    "\n    Checks if a given version and name of a cluster exists.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' postgres.cluster_exists '9.3'\n\n        salt '*' postgres.cluster_exists '9.3' 'main'\n    "
    return f'{version}/{name}' in cluster_list()

def cluster_remove(version, name='main', stop=False):
    if False:
        while True:
            i = 10
    "\n    Remove a cluster on a Postgres server. By default it doesn't try\n    to stop the cluster.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' postgres.cluster_remove '9.3'\n\n        salt '*' postgres.cluster_remove '9.3' 'main'\n\n        salt '*' postgres.cluster_remove '9.3' 'main' stop=True\n\n    "
    cmd = [salt.utils.path.which('pg_dropcluster')]
    if stop:
        cmd += ['--stop']
    cmd += [str(version), name]
    cmdstr = ' '.join([shlex.quote(c) for c in cmd])
    ret = __salt__['cmd.run_all'](cmdstr, python_shell=False)
    if ret.get('retcode', 0) != 0:
        log.error('Error removing a Postgresql cluster %s/%s', version, name)
    else:
        ret['changes'] = f'Successfully removed cluster {version}/{name}'
    return ret

def _parse_pg_lscluster(output):
    if False:
        i = 10
        return i + 15
    '\n    Helper function to parse the output of pg_lscluster\n    '
    cluster_dict = {}
    for line in output.splitlines():
        (version, name, port, status, user, datadir, log) = line.split()
        cluster_dict[f'{version}/{name}'] = {'port': int(port), 'status': status, 'user': user, 'datadir': datadir, 'log': log}
    return cluster_dict