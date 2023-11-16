"""
Control Linux Containers via Salt

:depends: lxc execution module
"""
import copy
import logging
import os
import time
import salt.client
import salt.key
import salt.utils.args
import salt.utils.cloud
import salt.utils.files
import salt.utils.stringutils
import salt.utils.virt
from salt.utils.odict import OrderedDict as _OrderedDict
log = logging.getLogger(__name__)
__func_alias__ = {'list_': 'list'}

def _do(name, fun, path=None):
    if False:
        while True:
            i = 10
    '\n    Invoke a function in the lxc module with no args\n\n    path\n        path to the container parent\n        default: /var/lib/lxc (system default)\n\n        .. versionadded:: 2015.8.0\n    '
    host = find_guest(name, quiet=True, path=path)
    if not host:
        return False
    with salt.client.get_local_client(__opts__['conf_file']) as client:
        cmd_ret = client.cmd_iter(host, 'lxc.{}'.format(fun), [name], kwarg={'path': path}, timeout=60)
        data = next(cmd_ret)
        data = data.get(host, {}).get('ret', None)
        if data:
            data = {host: data}
        return data

def _do_names(names, fun, path=None):
    if False:
        print('Hello World!')
    '\n    Invoke a function in the lxc module with no args\n\n    path\n        path to the container parent\n        default: /var/lib/lxc (system default)\n\n        .. versionadded:: 2015.8.0\n    '
    ret = {}
    hosts = find_guests(names, path=path)
    if not hosts:
        return False
    with salt.client.get_local_client(__opts__['conf_file']) as client:
        for (host, sub_names) in hosts.items():
            cmds = []
            for name in sub_names:
                cmds.append(client.cmd_iter(host, 'lxc.{}'.format(fun), [name], kwarg={'path': path}, timeout=60))
            for cmd in cmds:
                data = next(cmd)
                data = data.get(host, {}).get('ret', None)
                if data:
                    ret.update({host: data})
        return ret

def find_guest(name, quiet=False, path=None):
    if False:
        return 10
    '\n    Returns the host for a container.\n\n    path\n        path to the container parent\n        default: /var/lib/lxc (system default)\n\n        .. versionadded:: 2015.8.0\n\n\n    .. code-block:: bash\n\n        salt-run lxc.find_guest name\n    '
    if quiet:
        log.warning("'quiet' argument is being deprecated. Please migrate to --quiet")
    for data in _list_iter(path=path):
        (host, l) = next(iter(data.items()))
        for x in ('running', 'frozen', 'stopped'):
            if name in l[x]:
                if not quiet:
                    __jid_event__.fire_event({'data': host, 'outputter': 'lxc_find_host'}, 'progress')
                return host
    return None

def find_guests(names, path=None):
    if False:
        return 10
    '\n    Return a dict of hosts and named guests\n\n    path\n        path to the container parent\n        default: /var/lib/lxc (system default)\n\n        .. versionadded:: 2015.8.0\n\n    '
    ret = {}
    names = names.split(',')
    for data in _list_iter(path=path):
        (host, stat) = next(iter(data.items()))
        for state in stat:
            for name in stat[state]:
                if name in names:
                    if host in ret:
                        ret[host].append(name)
                    else:
                        ret[host] = [name]
    return ret

def init(names, host=None, saltcloud_mode=False, quiet=False, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Initialize a new container\n\n\n    .. code-block:: bash\n\n        salt-run lxc.init name host=minion_id [cpuset=cgroups_cpuset] \\\n                [cpushare=cgroups_cpushare] [memory=cgroups_memory] \\\n                [template=lxc_template_name] [clone=original name] \\\n                [profile=lxc_profile] [network_proflile=network_profile] \\\n                [nic=network_profile] [nic_opts=nic_opts] \\\n                [start=(true|false)] [seed=(true|false)] \\\n                [install=(true|false)] [config=minion_config] \\\n                [snapshot=(true|false)]\n\n    names\n        Name of the containers, supports a single name or a comma delimited\n        list of names.\n\n    host\n        Minion on which to initialize the container **(required)**\n\n    path\n        path to the container parent\n        default: /var/lib/lxc (system default)\n\n        .. versionadded:: 2015.8.0\n\n    saltcloud_mode\n        init the container with the saltcloud opts format instead\n        See lxc.init_interface module documentation\n\n    cpuset\n        cgroups cpuset.\n\n    cpushare\n        cgroups cpu shares.\n\n    memory\n        cgroups memory limit, in MB\n\n        .. versionchanged:: 2015.5.0\n            If no value is passed, no limit is set. In earlier Salt versions,\n            not passing this value causes a 1024MB memory limit to be set, and\n            it was necessary to pass ``memory=0`` to set no limit.\n\n    template\n        Name of LXC template on which to base this container\n\n    clone\n        Clone this container from an existing container\n\n    profile\n        A LXC profile (defined in config or pillar).\n\n    network_profile\n        Network profile to use for the container\n\n        .. versionadded:: 2015.5.2\n\n    nic\n        .. deprecated:: 2015.5.0\n            Use ``network_profile`` instead\n\n    nic_opts\n        Extra options for network interfaces. E.g.:\n\n        ``{"eth0": {"mac": "aa:bb:cc:dd:ee:ff", "ipv4": "10.1.1.1", "ipv6": "2001:db8::ff00:42:8329"}}``\n\n    start\n        Start the newly created container.\n\n    seed\n        Seed the container with the minion config and autosign its key.\n        Default: true\n\n    install\n        If salt-minion is not already installed, install it. Default: true\n\n    config\n        Optional config parameters. By default, the id is set to\n        the name of the container.\n    '
    path = kwargs.get('path', None)
    if quiet:
        log.warning("'quiet' argument is being deprecated. Please migrate to --quiet")
    ret = {'comment': '', 'result': True}
    if host is None:
        ret['comment'] = 'A host must be provided'
        ret['result'] = False
        return ret
    if isinstance(names, str):
        names = names.split(',')
    if not isinstance(names, list):
        ret['comment'] = 'Container names are not formed as a list'
        ret['result'] = False
        return ret
    alive = False
    with salt.client.get_local_client(__opts__['conf_file']) as client:
        try:
            if client.cmd(host, 'test.ping', timeout=20).get(host, None):
                alive = True
        except (TypeError, KeyError):
            pass
        if not alive:
            ret['comment'] = 'Host {} is not reachable'.format(host)
            ret['result'] = False
            return ret
        log.info('Searching for LXC Hosts')
        data = __salt__['lxc.list'](host, quiet=True, path=path)
        for (host, containers) in data.items():
            for name in names:
                if name in sum(containers.values(), []):
                    log.info("Container '%s' already exists on host '%s', init can be a NO-OP", name, host)
        if host not in data:
            ret['comment'] = "Host '{}' was not found".format(host)
            ret['result'] = False
            return ret
        kw = salt.utils.args.clean_kwargs(**kwargs)
        pub_key = kw.get('pub_key', None)
        priv_key = kw.get('priv_key', None)
        explicit_auth = pub_key and priv_key
        approve_key = kw.get('approve_key', True)
        seeds = {}
        seed_arg = kwargs.get('seed', True)
        if approve_key and (not explicit_auth):
            with salt.key.Key(__opts__) as skey:
                all_minions = skey.all_keys().get('minions', [])
                for name in names:
                    seed = seed_arg
                    if name in all_minions:
                        try:
                            if client.cmd(name, 'test.ping', timeout=20).get(name, None):
                                seed = False
                        except (TypeError, KeyError):
                            pass
                    seeds[name] = seed
                    kv = salt.utils.virt.VirtKey(host, name, __opts__)
                    if kv.authorize():
                        log.info('Container key will be preauthorized')
                    else:
                        ret['comment'] = 'Container key preauthorization failed'
                        ret['result'] = False
                        return ret
        log.info("Creating container(s) '%s' on host '%s'", names, host)
        cmds = []
        for name in names:
            args = [name]
            kw = salt.utils.args.clean_kwargs(**kwargs)
            if saltcloud_mode:
                kw = copy.deepcopy(kw)
                kw['name'] = name
                saved_kwargs = kw
                kw = client.cmd(host, 'lxc.cloud_init_interface', args + [kw], tgt_type='list', timeout=600).get(host, {})
                kw.update(saved_kwargs)
            name = kw.pop('name', name)
            kw['seed'] = seeds.get(name, seed_arg)
            if not kw['seed']:
                kw.pop('seed_cmd', '')
            cmds.append((host, name, client.cmd_iter(host, 'lxc.init', args, kwarg=kw, timeout=600)))
        done = ret.setdefault('done', [])
        errors = ret.setdefault('errors', _OrderedDict())
        for (ix, acmd) in enumerate(cmds):
            (hst, container_name, cmd) = acmd
            containers = ret.setdefault(hst, [])
            herrs = errors.setdefault(hst, _OrderedDict())
            serrs = herrs.setdefault(container_name, [])
            sub_ret = next(cmd)
            error = None
            if isinstance(sub_ret, dict) and host in sub_ret:
                j_ret = sub_ret[hst]
                container = j_ret.get('ret', {})
                if container and isinstance(container, dict):
                    if not container.get('result', False):
                        error = container
                else:
                    error = 'Invalid return for {}: {} {}'.format(container_name, container, sub_ret)
            else:
                error = sub_ret
                if not error:
                    error = 'unknown error (no return)'
            if error:
                ret['result'] = False
                serrs.append(error)
            else:
                container['container_name'] = name
                containers.append(container)
                done.append(container)
        ret['ping_status'] = bool(len(done))
        for container in done:
            container_name = container['container_name']
            key = os.path.join(__opts__['pki_dir'], 'minions', container_name)
            if explicit_auth:
                fcontent = ''
                if os.path.exists(key):
                    with salt.utils.files.fopen(key) as fic:
                        fcontent = salt.utils.stringutils.to_unicode(fic.read()).strip()
                pub_key = salt.utils.stringutils.to_unicode(pub_key)
                if pub_key.strip() != fcontent:
                    with salt.utils.files.fopen(key, 'w') as fic:
                        fic.write(salt.utils.stringutils.to_str(pub_key))
                        fic.flush()
            mid = j_ret.get('mid', None)
            if not mid:
                continue

            def testping(**kw):
                if False:
                    for i in range(10):
                        print('nop')
                mid_ = kw['mid']
                ping = client.cmd(mid_, 'test.ping', timeout=20)
                time.sleep(1)
                if ping:
                    return 'OK'
                raise Exception('Unresponsive {}'.format(mid_))
            ping = salt.utils.cloud.wait_for_fun(testping, timeout=21, mid=mid)
            if ping != 'OK':
                ret['ping_status'] = False
                ret['result'] = False
        if not done:
            ret['result'] = False
        if not quiet:
            __jid_event__.fire_event({'message': ret}, 'progress')
        return ret

def cloud_init(names, host=None, quiet=False, **kwargs):
    if False:
        return 10
    '\n    Wrapper for using lxc.init in saltcloud compatibility mode\n\n    names\n        Name of the containers, supports a single name or a comma delimited\n        list of names.\n\n    host\n        Minion to start the container on. Required.\n\n    path\n        path to the container parent\n        default: /var/lib/lxc (system default)\n\n        .. versionadded:: 2015.8.0\n\n    saltcloud_mode\n        init the container with the saltcloud opts format instead\n    '
    if quiet:
        log.warning("'quiet' argument is being deprecated. Please migrate to --quiet")
    return __salt__['lxc.init'](names=names, host=host, saltcloud_mode=True, quiet=quiet, **kwargs)

def _list_iter(host=None, path=None):
    if False:
        i = 10
        return i + 15
    '\n    Return a generator iterating over hosts\n\n    path\n        path to the container parent\n        default: /var/lib/lxc (system default)\n\n        .. versionadded:: 2015.8.0\n    '
    tgt = host or '*'
    with salt.client.get_local_client(__opts__['conf_file']) as client:
        for container_info in client.cmd_iter(tgt, 'lxc.list', kwarg={'path': path}):
            if not container_info:
                continue
            if not isinstance(container_info, dict):
                continue
            chunk = {}
            id_ = next(iter(container_info.keys()))
            if host and host != id_:
                continue
            if not isinstance(container_info[id_], dict):
                continue
            if 'ret' not in container_info[id_]:
                continue
            if not isinstance(container_info[id_]['ret'], dict):
                continue
            chunk[id_] = container_info[id_]['ret']
            yield chunk

def list_(host=None, quiet=False, path=None):
    if False:
        print('Hello World!')
    '\n    List defined containers (running, stopped, and frozen) for the named\n    (or all) host(s).\n\n    path\n        path to the container parent\n        default: /var/lib/lxc (system default)\n\n        .. versionadded:: 2015.8.0\n\n    .. code-block:: bash\n\n        salt-run lxc.list [host=minion_id]\n    '
    it = _list_iter(host, path=path)
    ret = {}
    for chunk in it:
        ret.update(chunk)
        if not quiet:
            __jid_event__.fire_event({'data': chunk, 'outputter': 'lxc_list'}, 'progress')
    return ret

def purge(name, delete_key=True, quiet=False, path=None):
    if False:
        print('Hello World!')
    '\n    Purge the named container and delete its minion key if present.\n    WARNING: Destroys all data associated with the container.\n\n    path\n        path to the container parent\n        default: /var/lib/lxc (system default)\n\n        .. versionadded:: 2015.8.0\n\n    .. code-block:: bash\n\n        salt-run lxc.purge name\n    '
    data = _do_names(name, 'destroy', path=path)
    if data is False:
        return data
    if delete_key:
        with salt.key.Key(__opts__) as skey:
            skey.delete_key(name)
    if data is None:
        return
    if not quiet:
        __jid_event__.fire_event({'data': data, 'outputter': 'lxc_purge'}, 'progress')
    return data

def start(name, quiet=False, path=None):
    if False:
        print('Hello World!')
    '\n    Start the named container.\n\n    path\n        path to the container parent\n        default: /var/lib/lxc (system default)\n\n        .. versionadded:: 2015.8.0\n\n    .. code-block:: bash\n\n        salt-run lxc.start name\n    '
    data = _do_names(name, 'start', path=path)
    if data and (not quiet):
        __jid_event__.fire_event({'data': data, 'outputter': 'lxc_start'}, 'progress')
    return data

def stop(name, quiet=False, path=None):
    if False:
        return 10
    '\n    Stop the named container.\n\n    path\n        path to the container parent\n        default: /var/lib/lxc (system default)\n\n        .. versionadded:: 2015.8.0\n\n    .. code-block:: bash\n\n        salt-run lxc.stop name\n    '
    data = _do_names(name, 'stop', path=path)
    if data and (not quiet):
        __jid_event__.fire_event({'data': data, 'outputter': 'lxc_force_off'}, 'progress')
    return data

def freeze(name, quiet=False, path=None):
    if False:
        while True:
            i = 10
    '\n    Freeze the named container\n\n    path\n        path to the container parent\n        default: /var/lib/lxc (system default)\n\n        .. versionadded:: 2015.8.0\n\n    .. code-block:: bash\n\n        salt-run lxc.freeze name\n    '
    data = _do_names(name, 'freeze')
    if data and (not quiet):
        __jid_event__.fire_event({'data': data, 'outputter': 'lxc_pause'}, 'progress')
    return data

def unfreeze(name, quiet=False, path=None):
    if False:
        i = 10
        return i + 15
    '\n    Unfreeze the named container\n\n    path\n        path to the container parent\n        default: /var/lib/lxc (system default)\n\n        .. versionadded:: 2015.8.0\n\n    .. code-block:: bash\n\n        salt-run lxc.unfreeze name\n    '
    data = _do_names(name, 'unfreeze', path=path)
    if data and (not quiet):
        __jid_event__.fire_event({'data': data, 'outputter': 'lxc_resume'}, 'progress')
    return data

def info(name, quiet=False, path=None):
    if False:
        i = 10
        return i + 15
    '\n    Returns information about a container.\n\n    path\n        path to the container parent\n        default: /var/lib/lxc (system default)\n\n        .. versionadded:: 2015.8.0\n\n    .. code-block:: bash\n\n        salt-run lxc.info name\n    '
    data = _do_names(name, 'info', path=path)
    if data and (not quiet):
        __jid_event__.fire_event({'data': data, 'outputter': 'lxc_info'}, 'progress')
    return data