"""
The Saltutil runner is used to sync custom types to the Master. See the
:mod:`saltutil module <salt.modules.saltutil>` for documentation on
managing updates to minions.

.. versionadded:: 2016.3.0
"""
import logging
import salt.utils.extmods
log = logging.getLogger(__name__)

def sync_all(saltenv='base', extmod_whitelist=None, extmod_blacklist=None):
    if False:
        while True:
            i = 10
    "\n    Sync all custom types\n\n    saltenv : base\n        The fileserver environment from which to sync. To sync from more than\n        one environment, pass a comma-separated list.\n\n    extmod_whitelist : None\n        dictionary of modules to sync based on type\n\n    extmod_blacklist : None\n        dictionary of modules to blacklist based on type\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-run saltutil.sync_all\n        salt-run saltutil.sync_all extmod_whitelist={'runners': ['custom_runner'], 'grains': []}\n    "
    log.debug('Syncing all')
    ret = {}
    ret['clouds'] = sync_clouds(saltenv=saltenv, extmod_whitelist=extmod_whitelist, extmod_blacklist=extmod_blacklist)
    ret['modules'] = sync_modules(saltenv=saltenv, extmod_whitelist=extmod_whitelist, extmod_blacklist=extmod_blacklist)
    ret['states'] = sync_states(saltenv=saltenv, extmod_whitelist=extmod_whitelist, extmod_blacklist=extmod_blacklist)
    ret['grains'] = sync_grains(saltenv=saltenv, extmod_whitelist=extmod_whitelist, extmod_blacklist=extmod_blacklist)
    ret['renderers'] = sync_renderers(saltenv=saltenv, extmod_whitelist=extmod_whitelist, extmod_blacklist=extmod_blacklist)
    ret['returners'] = sync_returners(saltenv=saltenv, extmod_whitelist=extmod_whitelist, extmod_blacklist=extmod_blacklist)
    ret['output'] = sync_output(saltenv=saltenv, extmod_whitelist=extmod_whitelist, extmod_blacklist=extmod_blacklist)
    ret['proxymodules'] = sync_proxymodules(saltenv=saltenv, extmod_whitelist=extmod_whitelist, extmod_blacklist=extmod_blacklist)
    ret['runners'] = sync_runners(saltenv=saltenv, extmod_whitelist=extmod_whitelist, extmod_blacklist=extmod_blacklist)
    ret['wheel'] = sync_wheel(saltenv=saltenv, extmod_whitelist=extmod_whitelist, extmod_blacklist=extmod_blacklist)
    ret['engines'] = sync_engines(saltenv=saltenv, extmod_whitelist=extmod_whitelist, extmod_blacklist=extmod_blacklist)
    ret['thorium'] = sync_thorium(saltenv=saltenv, extmod_whitelist=extmod_whitelist, extmod_blacklist=extmod_blacklist)
    ret['queues'] = sync_queues(saltenv=saltenv, extmod_whitelist=extmod_whitelist, extmod_blacklist=extmod_blacklist)
    ret['pillar'] = sync_pillar(saltenv=saltenv, extmod_whitelist=extmod_whitelist, extmod_blacklist=extmod_blacklist)
    ret['utils'] = sync_utils(saltenv=saltenv, extmod_whitelist=extmod_whitelist, extmod_blacklist=extmod_blacklist)
    ret['sdb'] = sync_sdb(saltenv=saltenv, extmod_whitelist=extmod_whitelist, extmod_blacklist=extmod_blacklist)
    ret['cache'] = sync_cache(saltenv=saltenv, extmod_whitelist=extmod_whitelist, extmod_blacklist=extmod_blacklist)
    ret['fileserver'] = sync_fileserver(saltenv=saltenv, extmod_whitelist=extmod_whitelist, extmod_blacklist=extmod_blacklist)
    ret['tops'] = sync_tops(saltenv=saltenv, extmod_whitelist=extmod_whitelist, extmod_blacklist=extmod_blacklist)
    ret['tokens'] = sync_eauth_tokens(saltenv=saltenv, extmod_whitelist=extmod_whitelist, extmod_blacklist=extmod_blacklist)
    ret['serializers'] = sync_serializers(saltenv=saltenv, extmod_whitelist=extmod_whitelist, extmod_blacklist=extmod_blacklist)
    ret['executors'] = sync_executors(saltenv=saltenv, extmod_whitelist=extmod_whitelist, extmod_blacklist=extmod_blacklist)
    ret['wrapper'] = sync_wrapper(saltenv=saltenv, extmod_whitelist=extmod_whitelist, extmod_blacklist=extmod_blacklist)
    ret['roster'] = sync_roster(saltenv=saltenv, extmod_whitelist=extmod_whitelist, extmod_blacklist=extmod_blacklist)
    return ret

def sync_modules(saltenv='base', extmod_whitelist=None, extmod_blacklist=None):
    if False:
        while True:
            i = 10
    '\n    Sync execution modules from ``salt://_modules`` to the master\n\n    saltenv : base\n        The fileserver environment from which to sync. To sync from more than\n        one environment, pass a comma-separated list.\n\n    extmod_whitelist : None\n        comma-separated list of modules to sync\n\n    extmod_blacklist : None\n        comma-separated list of modules to blacklist based on type\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-run saltutil.sync_modules\n    '
    return salt.utils.extmods.sync(__opts__, 'modules', saltenv=saltenv, extmod_whitelist=extmod_whitelist, extmod_blacklist=extmod_blacklist)[0]

def sync_states(saltenv='base', extmod_whitelist=None, extmod_blacklist=None):
    if False:
        while True:
            i = 10
    '\n    Sync state modules from ``salt://_states`` to the master\n\n    saltenv : base\n        The fileserver environment from which to sync. To sync from more than\n        one environment, pass a comma-separated list.\n\n    extmod_whitelist : None\n        comma-separated list of modules to sync\n\n    extmod_blacklist : None\n        comma-separated list of modules to blacklist based on type\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-run saltutil.sync_states\n    '
    return salt.utils.extmods.sync(__opts__, 'states', saltenv=saltenv, extmod_whitelist=extmod_whitelist, extmod_blacklist=extmod_blacklist)[0]

def sync_grains(saltenv='base', extmod_whitelist=None, extmod_blacklist=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Sync grains modules from ``salt://_grains`` to the master\n\n    saltenv : base\n        The fileserver environment from which to sync. To sync from more than\n        one environment, pass a comma-separated list.\n\n    extmod_whitelist : None\n        comma-separated list of modules to sync\n\n    extmod_blacklist : None\n        comma-separated list of modules to blacklist based on type\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-run saltutil.sync_grains\n    '
    return salt.utils.extmods.sync(__opts__, 'grains', saltenv=saltenv, extmod_whitelist=extmod_whitelist, extmod_blacklist=extmod_blacklist)[0]

def sync_renderers(saltenv='base', extmod_whitelist=None, extmod_blacklist=None):
    if False:
        i = 10
        return i + 15
    '\n    Sync renderer modules from from ``salt://_renderers`` to the master\n\n    saltenv : base\n        The fileserver environment from which to sync. To sync from more than\n        one environment, pass a comma-separated list.\n\n    extmod_whitelist : None\n        comma-separated list of modules to sync\n\n    extmod_blacklist : None\n        comma-separated list of modules to blacklist based on type\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-run saltutil.sync_renderers\n    '
    return salt.utils.extmods.sync(__opts__, 'renderers', saltenv=saltenv, extmod_whitelist=extmod_whitelist, extmod_blacklist=extmod_blacklist)[0]

def sync_returners(saltenv='base', extmod_whitelist=None, extmod_blacklist=None):
    if False:
        print('Hello World!')
    '\n    Sync returner modules from ``salt://_returners`` to the master\n\n    saltenv : base\n        The fileserver environment from which to sync. To sync from more than\n        one environment, pass a comma-separated list.\n\n    extmod_whitelist : None\n        comma-separated list of modules to sync\n\n    extmod_blacklist : None\n        comma-separated list of modules to blacklist based on type\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-run saltutil.sync_returners\n    '
    return salt.utils.extmods.sync(__opts__, 'returners', saltenv=saltenv, extmod_whitelist=extmod_whitelist, extmod_blacklist=extmod_blacklist)[0]

def sync_output(saltenv='base', extmod_whitelist=None, extmod_blacklist=None):
    if False:
        i = 10
        return i + 15
    '\n    Sync output modules from ``salt://_output`` to the master\n\n    saltenv : base\n        The fileserver environment from which to sync. To sync from more than\n        one environment, pass a comma-separated list.\n\n    extmod_whitelist : None\n        comma-separated list of modules to sync\n\n    extmod_blacklist : None\n        comma-separated list of modules to blacklist based on type\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-run saltutil.sync_output\n    '
    return salt.utils.extmods.sync(__opts__, 'output', saltenv=saltenv, extmod_whitelist=extmod_whitelist, extmod_blacklist=extmod_blacklist)[0]

def sync_proxymodules(saltenv='base', extmod_whitelist=None, extmod_blacklist=None):
    if False:
        while True:
            i = 10
    '\n    Sync proxy modules from ``salt://_proxy`` to the master\n\n    saltenv : base\n        The fileserver environment from which to sync. To sync from more than\n        one environment, pass a comma-separated list.\n\n    extmod_whitelist : None\n        comma-separated list of modules to sync\n\n    extmod_blacklist : None\n        comma-separated list of modules to blacklist based on type\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-run saltutil.sync_proxymodules\n    '
    return salt.utils.extmods.sync(__opts__, 'proxy', saltenv=saltenv, extmod_whitelist=extmod_whitelist, extmod_blacklist=extmod_blacklist)[0]

def sync_runners(saltenv='base', extmod_whitelist=None, extmod_blacklist=None):
    if False:
        return 10
    '\n    Sync runners from ``salt://_runners`` to the master\n\n    saltenv : base\n        The fileserver environment from which to sync. To sync from more than\n        one environment, pass a comma-separated list.\n\n    extmod_whitelist : None\n        comma-separated list of modules to sync\n\n    extmod_blacklist : None\n        comma-separated list of modules to blacklist based on type\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-run saltutil.sync_runners\n    '
    return salt.utils.extmods.sync(__opts__, 'runners', saltenv=saltenv, extmod_whitelist=extmod_whitelist, extmod_blacklist=extmod_blacklist)[0]

def sync_wheel(saltenv='base', extmod_whitelist=None, extmod_blacklist=None):
    if False:
        print('Hello World!')
    '\n    Sync wheel modules from ``salt://_wheel`` to the master\n\n    saltenv : base\n        The fileserver environment from which to sync. To sync from more than\n        one environment, pass a comma-separated list.\n\n    extmod_whitelist : None\n        comma-separated list of modules to sync\n\n    extmod_blacklist : None\n        comma-separated list of modules to blacklist based on type\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-run saltutil.sync_wheel\n    '
    return salt.utils.extmods.sync(__opts__, 'wheel', saltenv=saltenv, extmod_whitelist=extmod_whitelist, extmod_blacklist=extmod_blacklist)[0]

def sync_engines(saltenv='base', extmod_whitelist=None, extmod_blacklist=None):
    if False:
        i = 10
        return i + 15
    '\n    Sync engines from ``salt://_engines`` to the master\n\n    saltenv : base\n        The fileserver environment from which to sync. To sync from more than\n        one environment, pass a comma-separated list.\n\n    extmod_whitelist : None\n        comma-separated list of modules to sync\n\n    extmod_blacklist : None\n        comma-separated list of modules to blacklist based on type\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-run saltutil.sync_engines\n    '
    return salt.utils.extmods.sync(__opts__, 'engines', saltenv=saltenv, extmod_whitelist=extmod_whitelist, extmod_blacklist=extmod_blacklist)[0]

def sync_thorium(saltenv='base', extmod_whitelist=None, extmod_blacklist=None):
    if False:
        return 10
    '\n    .. versionadded:: 2018.3.0\n\n    Sync Thorium from ``salt://_thorium`` to the master\n\n    saltenv: ``base``\n        The fileserver environment from which to sync. To sync from more than\n        one environment, pass a comma-separated list.\n\n    extmod_whitelist\n        comma-separated list of modules to sync\n\n    extmod_blacklist\n        comma-separated list of modules to blacklist based on type\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-run saltutil.sync_thorium\n    '
    return salt.utils.extmods.sync(__opts__, 'thorium', saltenv=saltenv, extmod_whitelist=extmod_whitelist, extmod_blacklist=extmod_blacklist)[0]

def sync_queues(saltenv='base', extmod_whitelist=None, extmod_blacklist=None):
    if False:
        return 10
    '\n    Sync queue modules from ``salt://_queues`` to the master\n\n    saltenv : base\n        The fileserver environment from which to sync. To sync from more than\n        one environment, pass a comma-separated list.\n\n    extmod_whitelist : None\n        comma-separated list of modules to sync\n\n    extmod_blacklist : None\n        comma-separated list of modules to blacklist based on type\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-run saltutil.sync_queues\n    '
    return salt.utils.extmods.sync(__opts__, 'queues', saltenv=saltenv, extmod_whitelist=extmod_whitelist, extmod_blacklist=extmod_blacklist)[0]

def sync_pillar(saltenv='base', extmod_whitelist=None, extmod_blacklist=None):
    if False:
        while True:
            i = 10
    '\n    Sync pillar modules from ``salt://_pillar`` to the master\n\n    saltenv : base\n        The fileserver environment from which to sync. To sync from more than\n        one environment, pass a comma-separated list.\n\n    extmod_whitelist : None\n        comma-separated list of modules to sync\n\n    extmod_blacklist : None\n        comma-separated list of modules to blacklist based on type\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-run saltutil.sync_pillar\n    '
    return salt.utils.extmods.sync(__opts__, 'pillar', saltenv=saltenv, extmod_whitelist=extmod_whitelist, extmod_blacklist=extmod_blacklist)[0]

def sync_utils(saltenv='base', extmod_whitelist=None, extmod_blacklist=None):
    if False:
        return 10
    '\n    .. versionadded:: 2016.11.0\n\n    Sync utils modules from ``salt://_utils`` to the master\n\n    saltenv : base\n        The fileserver environment from which to sync. To sync from more than\n        one environment, pass a comma-separated list.\n\n    extmod_whitelist : None\n        comma-separated list of modules to sync\n\n    extmod_blacklist : None\n        comma-separated list of modules to blacklist based on type\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-run saltutil.sync_utils\n    '
    return salt.utils.extmods.sync(__opts__, 'utils', saltenv=saltenv, extmod_whitelist=extmod_whitelist, extmod_blacklist=extmod_blacklist)[0]

def sync_sdb(saltenv='base', extmod_whitelist=None, extmod_blacklist=None):
    if False:
        while True:
            i = 10
    '\n    .. versionadded:: 2017.7.0\n\n    Sync sdb modules from ``salt://_sdb`` to the master\n\n    saltenv : base\n        The fileserver environment from which to sync. To sync from more than\n        one environment, pass a comma-separated list.\n\n    extmod_whitelist : None\n        comma-separated list of modules to sync\n\n    extmod_blacklist : None\n        comma-separated list of modules to blacklist based on type\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-run saltutil.sync_sdb\n    '
    return salt.utils.extmods.sync(__opts__, 'sdb', saltenv=saltenv, extmod_whitelist=extmod_whitelist, extmod_blacklist=extmod_blacklist)[0]

def sync_tops(saltenv='base', extmod_whitelist=None, extmod_blacklist=None):
    if False:
        i = 10
        return i + 15
    '\n    .. versionadded:: 2016.3.7,2016.11.4,2017.7.0\n\n    Sync master_tops modules from ``salt://_tops`` to the master\n\n    saltenv : base\n        The fileserver environment from which to sync. To sync from more than\n        one environment, pass a comma-separated list.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-run saltutil.sync_tops\n    '
    return salt.utils.extmods.sync(__opts__, 'tops', saltenv=saltenv, extmod_whitelist=extmod_whitelist, extmod_blacklist=extmod_blacklist)[0]

def sync_cache(saltenv='base', extmod_whitelist=None, extmod_blacklist=None):
    if False:
        return 10
    '\n    .. versionadded:: 2017.7.0\n\n    Sync cache modules from ``salt://_cache`` to the master\n\n    saltenv : base\n        The fileserver environment from which to sync. To sync from more than\n        one environment, pass a comma-separated list.\n\n    extmod_whitelist : None\n        comma-separated list of modules to sync\n\n    extmod_blacklist : None\n        comma-separated list of modules to blacklist based on type\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-run saltutil.sync_cache\n    '
    return salt.utils.extmods.sync(__opts__, 'cache', saltenv=saltenv, extmod_whitelist=extmod_whitelist, extmod_blacklist=extmod_blacklist)[0]

def sync_fileserver(saltenv='base', extmod_whitelist=None, extmod_blacklist=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    .. versionadded:: 2018.3.0\n\n    Sync fileserver modules from ``salt://_fileserver`` to the master\n\n    saltenv : base\n        The fileserver environment from which to sync. To sync from more than\n        one environment, pass a comma-separated list.\n\n    extmod_whitelist : None\n        comma-separated list of modules to sync\n\n    extmod_blacklist : None\n        comma-separated list of modules to blacklist based on type\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-run saltutil.sync_fileserver\n    '
    return salt.utils.extmods.sync(__opts__, 'fileserver', saltenv=saltenv, extmod_whitelist=extmod_whitelist, extmod_blacklist=extmod_blacklist)[0]

def sync_clouds(saltenv='base', extmod_whitelist=None, extmod_blacklist=None):
    if False:
        while True:
            i = 10
    '\n    .. versionadded:: 2017.7.0\n\n    Sync cloud modules from ``salt://_clouds`` to the master\n\n    saltenv : base\n        The fileserver environment from which to sync. To sync from more than\n        one environment, pass a comma-separated list.\n\n    extmod_whitelist : None\n        comma-separated list of modules to sync\n\n    extmod_blacklist : None\n        comma-separated list of modules to blacklist based on type\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-run saltutil.sync_clouds\n    '
    return salt.utils.extmods.sync(__opts__, 'clouds', saltenv=saltenv, extmod_whitelist=extmod_whitelist, extmod_blacklist=extmod_blacklist)[0]

def sync_roster(saltenv='base', extmod_whitelist=None, extmod_blacklist=None):
    if False:
        i = 10
        return i + 15
    '\n    .. versionadded:: 2017.7.0\n\n    Sync roster modules from ``salt://_roster`` to the master\n\n    saltenv : base\n        The fileserver environment from which to sync. To sync from more than\n        one environment, pass a comma-separated list.\n\n    extmod_whitelist : None\n        comma-separated list of modules to sync\n\n    extmod_blacklist : None\n        comma-separated list of modules to blacklist based on type\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-run saltutil.sync_roster\n    '
    return salt.utils.extmods.sync(__opts__, 'roster', saltenv=saltenv, extmod_whitelist=extmod_whitelist, extmod_blacklist=extmod_blacklist)[0]

def sync_eauth_tokens(saltenv='base', extmod_whitelist=None, extmod_blacklist=None):
    if False:
        i = 10
        return i + 15
    '\n    .. versionadded:: 2018.3.0\n\n    Sync eauth token modules from ``salt://_tokens`` to the master\n\n    saltenv : base\n        The fileserver environment from which to sync. To sync from more than\n        one environment, pass a comma-separated list.\n\n    extmod_whitelist : None\n        comma-separated list of modules to sync\n\n    extmod_blacklist : None\n        comma-separated list of modules to blacklist based on type\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-run saltutil.sync_eauth_tokens\n    '
    return salt.utils.extmods.sync(__opts__, 'tokens', saltenv=saltenv, extmod_whitelist=extmod_whitelist, extmod_blacklist=extmod_blacklist)[0]

def sync_serializers(saltenv='base', extmod_whitelist=None, extmod_blacklist=None):
    if False:
        return 10
    '\n    .. versionadded:: 2019.2.0\n\n    Sync serializer modules from ``salt://_serializers`` to the master\n\n    saltenv : base\n        The fileserver environment from which to sync. To sync from more than\n        one environment, pass a comma-separated list.\n\n    extmod_whitelist : None\n        comma-seperated list of modules to sync\n\n    extmod_blacklist : None\n        comma-seperated list of modules to blacklist based on type\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-run saltutil.sync_utils\n    '
    return salt.utils.extmods.sync(__opts__, 'serializers', saltenv=saltenv, extmod_whitelist=extmod_whitelist, extmod_blacklist=extmod_blacklist)[0]

def sync_executors(saltenv='base', extmod_whitelist=None, extmod_blacklist=None):
    if False:
        i = 10
        return i + 15
    '\n    .. versionadded:: 3000\n\n    Sync executor modules from ``salt://_executors`` to the master\n\n    saltenv : base\n        The fileserver environment from which to sync. To sync from more than\n        one environment, pass a comma-separated list.\n\n    extmod_whitelist : None\n        comma-seperated list of modules to sync\n\n    extmod_blacklist : None\n        comma-seperated list of modules to blacklist based on type\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-run saltutil.sync_executors\n    '
    return salt.utils.extmods.sync(__opts__, 'executors', saltenv=saltenv, extmod_whitelist=extmod_whitelist, extmod_blacklist=extmod_blacklist)[0]

def sync_wrapper(saltenv='base', extmod_whitelist=None, extmod_blacklist=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    .. versionadded:: 3007.0\n\n    Sync salt-ssh wrapper modules from ``salt://_wrapper`` to the master.\n\n    saltenv : base\n        The fileserver environment from which to sync. To sync from more than\n        one environment, pass a comma-separated list.\n\n    extmod_whitelist : None\n        comma-seperated list of modules to sync\n\n    extmod_blacklist : None\n        comma-seperated list of modules to blacklist based on type\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-run saltutil.sync_wrapper\n    '
    return salt.utils.extmods.sync(__opts__, 'wrapper', saltenv=saltenv, extmod_whitelist=extmod_whitelist, extmod_blacklist=extmod_blacklist)[0]