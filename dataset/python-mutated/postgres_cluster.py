"""
Management of PostgreSQL clusters
=================================

The postgres_cluster state module is used to manage PostgreSQL clusters.
Clusters can be set as either absent or present

.. code-block:: yaml

    create cluster 9.3 main:
      postgres_cluster.present:
          - name: 'main'
          - version: '9.3'
"""

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Only load if the deb_postgres module is present\n    '
    if 'postgres.cluster_exists' not in __salt__:
        return (False, 'Unable to load postgres module.  Make sure `postgres.bins_dir` is set.')
    return True

def present(version, name, port=None, encoding=None, locale=None, datadir=None, allow_group_access=None, data_checksums=None, wal_segsize=None):
    if False:
        while True:
            i = 10
    '\n    Ensure that the named cluster is present with the specified properties.\n    For more information about all of these options see man pg_createcluster(1)\n\n    version\n        Version of the postgresql cluster\n\n    name\n        The name of the cluster\n\n    port\n        Cluster port\n\n    encoding\n        The character encoding scheme to be used in this database\n\n    locale\n        Locale with which to create cluster\n\n    datadir\n        Where the cluster is stored\n\n    allow_group_access\n        Allows users in the same group as the cluster owner to read all cluster files created by initdb\n\n    data_checksums\n        Use checksums on data pages\n\n    wal_segsize\n        Set the WAL segment size, in megabytes\n\n        .. versionadded:: 2016.3.0\n    '
    msg = 'Cluster {}/{} is already present'.format(version, name)
    ret = {'name': name, 'changes': {}, 'result': True, 'comment': msg}
    if __salt__['postgres.cluster_exists'](version, name):
        infos = __salt__['postgres.cluster_list'](verbose=True)
        info = infos['{}/{}'.format(version, name)]
        if any((port != info['port'] if port else False, datadir != info['datadir'] if datadir else False)):
            ret['comment'] = "Cluster {}/{} has wrong parameters which couldn't be changed on fly.".format(version, name)
            ret['result'] = False
        return ret
    if __opts__.get('test'):
        ret['result'] = None
        msg = 'Cluster {0}/{1} is set to be created'
        ret['comment'] = msg.format(version, name)
        return ret
    cluster = __salt__['postgres.cluster_create'](version=version, name=name, port=port, locale=locale, encoding=encoding, datadir=datadir, allow_group_access=allow_group_access, data_checksums=data_checksums, wal_segsize=wal_segsize)
    if cluster:
        msg = 'The cluster {0}/{1} has been created'
        ret['comment'] = msg.format(version, name)
        ret['changes']['{}/{}'.format(version, name)] = 'Present'
    else:
        msg = 'Failed to create cluster {0}/{1}'
        ret['comment'] = msg.format(version, name)
        ret['result'] = False
    return ret

def absent(version, name):
    if False:
        while True:
            i = 10
    '\n    Ensure that the named cluster is absent\n\n    version\n        Version of the postgresql server of the cluster to remove\n\n    name\n        The name of the cluster to remove\n\n        .. versionadded:: 2016.3.0\n    '
    ret = {'name': name, 'changes': {}, 'result': True, 'comment': ''}
    if __salt__['postgres.cluster_exists'](version, name):
        if __opts__.get('test'):
            ret['result'] = None
            msg = 'Cluster {0}/{1} is set to be removed'
            ret['comment'] = msg.format(version, name)
            return ret
        if __salt__['postgres.cluster_remove'](version, name, True):
            msg = 'Cluster {0}/{1} has been removed'
            ret['comment'] = msg.format(version, name)
            ret['changes'][name] = 'Absent'
            return ret
    ret['comment'] = 'Cluster {}/{} is not present, so it cannot be removed'.format(version, name)
    return ret