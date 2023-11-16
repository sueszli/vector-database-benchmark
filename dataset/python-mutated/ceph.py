"""
Module to provide ceph control with salt.

:depends:   - ceph_cfg Python module

.. versionadded:: 2016.11.0
"""
import logging
log = logging.getLogger(__name__)
__virtualname__ = 'ceph'
try:
    import ceph_cfg
    HAS_CEPH_CFG = True
except ImportError:
    HAS_CEPH_CFG = False

def __virtual__():
    if False:
        print('Hello World!')
    if HAS_CEPH_CFG is False:
        msg = 'ceph_cfg unavailable: {} execution module cant be loaded '.format(__virtualname__)
        return (False, msg)
    return __virtualname__

def partition_list():
    if False:
        print('Hello World!')
    "\n    List partitions by disk\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ceph.partition_list\n    "
    return ceph_cfg.partition_list()

def partition_list_osd():
    if False:
        i = 10
        return i + 15
    "\n    List all OSD data partitions by partition\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ceph.partition_list_osd\n    "
    return ceph_cfg.partition_list_osd()

def partition_list_journal():
    if False:
        for i in range(10):
            print('nop')
    "\n    List all OSD journal partitions by partition\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ceph.partition_list_journal\n    "
    return ceph_cfg.partition_list_journal()

def osd_discover():
    if False:
        i = 10
        return i + 15
    "\n    List all OSD by cluster\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ceph.osd_discover\n\n    "
    return ceph_cfg.osd_discover()

def partition_is(dev):
    if False:
        print('Hello World!')
    "\n    Check whether a given device path is a partition or a full disk.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ceph.partition_is /dev/sdc1\n    "
    return ceph_cfg.partition_is(dev)

def zap(target=None, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Destroy the partition table and content of a given disk.\n\n    .. code-block:: bash\n\n        salt '*' ceph.osd_prepare 'dev'='/dev/vdc' \\\n                'cluster_name'='ceph' \\\n                'cluster_uuid'='cluster_uuid'\n\n    dev\n        The block device to format.\n\n    cluster_name\n        The cluster name. Defaults to ``ceph``.\n\n    cluster_uuid\n        The cluster UUID. Defaults to value found in ceph config file.\n    "
    if target is not None:
        log.warning('Deprecated use of function, use kwargs')
    target = kwargs.get('dev', target)
    kwargs['dev'] = target
    return ceph_cfg.zap(**kwargs)

def osd_prepare(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Prepare an OSD\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' ceph.osd_prepare \'osd_dev\'=\'/dev/vdc\' \\\n                \'journal_dev\'=\'device\' \\\n                \'cluster_name\'=\'ceph\' \\\n                \'cluster_uuid\'=\'cluster_uuid\' \\\n                \'osd_fs_type\'=\'xfs\' \\\n                \'osd_uuid\'=\'2a143b73-6d85-4389-a9e9-b8a78d9e1e07\' \\\n                \'journal_uuid\'=\'4562a5db-ff6f-4268-811d-12fd4a09ae98\'\n\n    cluster_uuid\n        The device to store the osd data on.\n\n    journal_dev\n        The journal device. defaults to osd_dev.\n\n    cluster_name\n        The cluster name. Defaults to ``ceph``.\n\n    cluster_uuid\n        The cluster date will be added too. Defaults to the value found in local config.\n\n    osd_fs_type\n        set the file system to store OSD data with. Defaults to "xfs".\n\n    osd_uuid\n        set the OSD data UUID. If set will return if OSD with data UUID already exists.\n\n    journal_uuid\n        set the OSD journal UUID. If set will return if OSD with journal UUID already exists.\n    '
    return ceph_cfg.osd_prepare(**kwargs)

def osd_activate(**kwargs):
    if False:
        return 10
    "\n    Activate an OSD\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ceph.osd_activate 'osd_dev'='/dev/vdc'\n    "
    return ceph_cfg.osd_activate(**kwargs)

def keyring_create(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Create keyring for cluster\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ceph.keyring_create \\\n                'keyring_type'='admin' \\\n                'cluster_name'='ceph' \\\n                'cluster_uuid'='cluster_uuid'\n\n    keyring_type (required)\n        One of ``admin``, ``mon``, ``osd``, ``rgw``, ``mds``\n\n    cluster_uuid\n        The cluster UUID. Defaults to value found in ceph config file.\n\n    cluster_name\n        The cluster name. Defaults to ``ceph``.\n    "
    return ceph_cfg.keyring_create(**kwargs)

def keyring_save(**kwargs):
    if False:
        while True:
            i = 10
    "\n    Create save keyring locally\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ceph.keyring_save \\\n                'keyring_type'='admin' \\\n                'cluster_name'='ceph' \\\n                'cluster_uuid'='cluster_uuid'\n\n    keyring_type (required)\n        One of ``admin``, ``mon``, ``osd``, ``rgw``, ``mds``\n\n    cluster_uuid\n        The cluster UUID. Defaults to value found in ceph config file.\n\n    cluster_name\n        The cluster name. Defaults to ``ceph``.\n    "
    return ceph_cfg.keyring_save(**kwargs)

def keyring_purge(**kwargs):
    if False:
        print('Hello World!')
    "\n    Delete keyring for cluster\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ceph.keyring_purge \\\n                'keyring_type'='admin' \\\n                'cluster_name'='ceph' \\\n                'cluster_uuid'='cluster_uuid'\n\n    keyring_type (required)\n        One of ``admin``, ``mon``, ``osd``, ``rgw``, ``mds``\n\n    cluster_uuid\n        The cluster UUID. Defaults to value found in ceph config file.\n\n    cluster_name\n        The cluster name. Defaults to ``ceph``.\n\n    If no ceph config file is found, this command will fail.\n    "
    return ceph_cfg.keyring_purge(**kwargs)

def keyring_present(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns ``True`` if the keyring is present on disk, otherwise ``False``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ceph.keyring_present \\\n                'keyring_type'='admin' \\\n                'cluster_name'='ceph' \\\n                'cluster_uuid'='cluster_uuid'\n\n    keyring_type (required)\n        One of ``admin``, ``mon``, ``osd``, ``rgw``, ``mds``\n\n    cluster_uuid\n        The cluster UUID. Defaults to value found in ceph config file.\n\n    cluster_name\n        The cluster name. Defaults to ``ceph``.\n    "
    return ceph_cfg.keyring_present(**kwargs)

def keyring_auth_add(**kwargs):
    if False:
        print('Hello World!')
    "\n    Add keyring to authorized list\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ceph.keyring_auth_add \\\n                'keyring_type'='admin' \\\n                'cluster_name'='ceph' \\\n                'cluster_uuid'='cluster_uuid'\n\n    keyring_type (required)\n        One of ``admin``, ``mon``, ``osd``, ``rgw``, ``mds``\n\n    cluster_uuid\n        The cluster UUID. Defaults to value found in ceph config file.\n\n    cluster_name\n        The cluster name. Defaults to ``ceph``.\n    "
    return ceph_cfg.keyring_auth_add(**kwargs)

def keyring_auth_del(**kwargs):
    if False:
        return 10
    "\n    Remove keyring from authorised list\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ceph.keyring_osd_auth_del \\\n                'keyring_type'='admin' \\\n                'cluster_name'='ceph' \\\n                'cluster_uuid'='cluster_uuid'\n\n    keyring_type (required)\n        One of ``admin``, ``mon``, ``osd``, ``rgw``, ``mds``\n\n    cluster_uuid\n        The cluster UUID. Defaults to value found in ceph config file.\n\n    cluster_name\n        The cluster name. Defaults to ``ceph``.\n    "
    return ceph_cfg.keyring_auth_del(**kwargs)

def mon_is(**kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Returns ``True`` if the target is a mon node, otherwise ``False``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ceph.mon_is \\\n                'cluster_name'='ceph' \\\n                'cluster_uuid'='cluster_uuid'\n\n    cluster_name\n        The cluster name. Defaults to ``ceph``.\n\n    cluster_uuid\n        The cluster UUID. Defaults to value found in ceph config file.\n    "
    return ceph_cfg.mon_is(**kwargs)

def mon_status(**kwargs):
    if False:
        return 10
    "\n    Get status from mon daemon\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ceph.mon_status \\\n                'cluster_name'='ceph' \\\n                'cluster_uuid'='cluster_uuid'\n\n    cluster_uuid\n        The cluster UUID. Defaults to value found in ceph config file.\n\n    cluster_name\n        The cluster name. Defaults to ``ceph``.\n    "
    return ceph_cfg.status(**kwargs)

def mon_quorum(**kwargs):
    if False:
        while True:
            i = 10
    "\n    Returns ``True`` if the mon daemon is in the quorum, otherwise ``False``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ceph.mon_quorum \\\n                'cluster_name'='ceph' \\\n                'cluster_uuid'='cluster_uuid'\n\n    cluster_uuid\n        The cluster UUID. Defaults to value found in ceph config file.\n\n    cluster_name\n        The cluster name. Defaults to ``ceph``.\n    "
    return ceph_cfg.mon_quorum(**kwargs)

def mon_active(**kwargs):
    if False:
        return 10
    "\n    Returns ``True`` if the mon daemon is running, otherwise ``False``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ceph.mon_active \\\n                'cluster_name'='ceph' \\\n                'cluster_uuid'='cluster_uuid'\n\n    cluster_uuid\n        The cluster UUID. Defaults to value found in ceph config file.\n\n    cluster_name\n        The cluster name. Defaults to ``ceph``.\n    "
    return ceph_cfg.mon_active(**kwargs)

def mon_create(**kwargs):
    if False:
        while True:
            i = 10
    "\n    Create a mon node\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ceph.mon_create \\\n                'cluster_name'='ceph' \\\n                'cluster_uuid'='cluster_uuid'\n\n    cluster_uuid\n        The cluster UUID. Defaults to value found in ceph config file.\n\n    cluster_name\n        The cluster name. Defaults to ``ceph``.\n    "
    return ceph_cfg.mon_create(**kwargs)

def rgw_pools_create(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Create pools for rgw\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ceph.rgw_pools_create\n\n    cluster_uuid\n        The cluster UUID. Defaults to value found in ceph config file.\n\n    cluster_name\n        The cluster name. Defaults to ``ceph``.\n    "
    return ceph_cfg.rgw_pools_create(**kwargs)

def rgw_pools_missing(**kwargs):
    if False:
        print('Hello World!')
    "\n    Show pools missing for rgw\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ceph.rgw_pools_missing\n\n    cluster_uuid\n        The cluster UUID. Defaults to value found in ceph config file.\n\n    cluster_name\n        The cluster name. Defaults to ``ceph``.\n    "
    return ceph_cfg.rgw_pools_missing(**kwargs)

def rgw_create(**kwargs):
    if False:
        return 10
    "\n    Create a rgw\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ceph.rgw_create \\\n                'name' = 'rgw.name' \\\n                'cluster_name'='ceph' \\\n                'cluster_uuid'='cluster_uuid'\n\n    name (required)\n        The RGW client name. Must start with ``rgw.``\n\n    cluster_uuid\n        The cluster UUID. Defaults to value found in ceph config file.\n\n    cluster_name\n        The cluster name. Defaults to ``ceph``.\n    "
    return ceph_cfg.rgw_create(**kwargs)

def rgw_destroy(**kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Remove a rgw\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ceph.rgw_destroy \\\n                'name' = 'rgw.name' \\\n                'cluster_name'='ceph' \\\n                'cluster_uuid'='cluster_uuid'\n\n    name (required)\n        The RGW client name (must start with ``rgw.``)\n\n    cluster_uuid\n        The cluster UUID. Defaults to value found in ceph config file.\n\n    cluster_name\n        The cluster name. Defaults to ``ceph``.\n    "
    return ceph_cfg.rgw_destroy(**kwargs)

def mds_create(**kwargs):
    if False:
        return 10
    "\n    Create a mds\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ceph.mds_create \\\n                'name' = 'mds.name' \\\n                'port' = 1000, \\\n                'addr' = 'fqdn.example.org' \\\n                'cluster_name'='ceph' \\\n                'cluster_uuid'='cluster_uuid'\n\n    name (required)\n        The MDS name (must start with ``mds.``)\n\n    port (required)\n        Port to which the MDS will listen\n\n    addr (required)\n        Address or IP address for the MDS to listen\n\n    cluster_uuid\n        The cluster UUID. Defaults to value found in ceph config file.\n\n    cluster_name\n        The cluster name. Defaults to ``ceph``.\n    "
    return ceph_cfg.mds_create(**kwargs)

def mds_destroy(**kwargs):
    if False:
        return 10
    "\n    Remove a mds\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ceph.mds_destroy \\\n                'name' = 'mds.name' \\\n                'cluster_name'='ceph' \\\n                'cluster_uuid'='cluster_uuid'\n\n    name (required)\n        The MDS name (must start with ``mds.``)\n\n    cluster_uuid\n        The cluster UUID. Defaults to value found in ceph config file.\n\n    cluster_name\n        The cluster name. Defaults to ``ceph``.\n    "
    return ceph_cfg.mds_destroy(**kwargs)

def keyring_auth_list(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    List all cephx authorization keys\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ceph.keyring_auth_list \\\n                'cluster_name'='ceph' \\\n                'cluster_uuid'='cluster_uuid'\n\n    cluster_name\n        The cluster name. Defaults to ``ceph``.\n\n    cluster_uuid\n        The cluster UUID. Defaults to value found in ceph config file.\n    "
    return ceph_cfg.keyring_auth_list(**kwargs)

def pool_list(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    List all pools\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ceph.pool_list \\\n                'cluster_name'='ceph' \\\n                'cluster_uuid'='cluster_uuid'\n\n    cluster_name\n        The cluster name. Defaults to ``ceph``.\n\n    cluster_uuid\n        The cluster UUID. Defaults to value found in ceph config file.\n    "
    return ceph_cfg.pool_list(**kwargs)

def pool_add(pool_name, **kwargs):
    if False:
        return 10
    '\n    Create a pool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' ceph.pool_add pool_name \\\n                \'cluster_name\'=\'ceph\' \\\n                \'cluster_uuid\'=\'cluster_uuid\'\n\n    cluster_name\n        The cluster name. Defaults to ``ceph``.\n\n    cluster_uuid\n        The cluster UUID. Defaults to value found in ceph config file.\n\n    pg_num\n        Default to 8\n\n    pgp_num\n        Default to pg_num\n\n    pool_type\n        can take values "replicated" or "erasure"\n\n    erasure_code_profile\n        The "erasure_code_profile"\n\n    crush_ruleset\n        The crush map rule set\n    '
    return ceph_cfg.pool_add(pool_name, **kwargs)

def pool_del(pool_name, **kwargs):
    if False:
        print('Hello World!')
    "\n    Delete a pool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ceph.pool_del pool_name \\\n                'cluster_name'='ceph' \\\n                'cluster_uuid'='cluster_uuid'\n\n    cluster_name\n        The cluster name. Defaults to ``ceph``.\n\n    cluster_uuid\n        The cluster UUID. Defaults to value found in ceph config file.\n    "
    return ceph_cfg.pool_del(pool_name, **kwargs)

def purge(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    purge ceph configuration on the node\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ceph.purge \\\n                'cluster_name'='ceph' \\\n                'cluster_uuid'='cluster_uuid'\n\n    cluster_name\n        The cluster name. Defaults to ``ceph``.\n\n    cluster_uuid\n        The cluster UUID. Defaults to value found in ceph config file.\n    "
    return ceph_cfg.purge(**kwargs)

def ceph_version():
    if False:
        for i in range(10):
            print('nop')
    "\n    Get the version of ceph installed\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ceph.ceph_version\n    "
    return ceph_cfg.ceph_version()

def cluster_quorum(**kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Get the cluster's quorum status\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ceph.cluster_quorum \\\n                'cluster_name'='ceph' \\\n                'cluster_uuid'='cluster_uuid'\n\n    cluster_uuid\n        The cluster UUID. Defaults to value found in ceph config file.\n\n    cluster_name\n        The cluster name. Defaults to ``ceph``.\n    "
    return ceph_cfg.cluster_quorum(**kwargs)

def cluster_status(**kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Get the cluster status, including health if in quorum\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ceph.cluster_status \\\n                'cluster_name'='ceph' \\\n                'cluster_uuid'='cluster_uuid'\n\n    cluster_uuid\n        The cluster UUID. Defaults to value found in ceph config file.\n\n    cluster_name\n        The cluster name. Defaults to ``ceph``.\n    "
    return ceph_cfg.cluster_status(**kwargs)