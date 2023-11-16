"""
Run munin plugins/checks from salt and format the output as data.
"""
import os
import stat
import salt.utils.files
import salt.utils.stringutils
PLUGINDIR = '/etc/munin/plugins/'

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Only load the module if munin-node is installed\n    '
    if os.path.exists('/etc/munin/munin-node.conf'):
        return 'munin'
    return (False, 'The munin execution module cannot be loaded: munin-node is not installed.')

def _get_conf(fname='/etc/munin/munin-node.cfg'):
    if False:
        return 10
    with salt.utils.files.fopen(fname, 'r') as fp_:
        return salt.utils.stringutils.to_unicode(fp_.read())

def run(plugins):
    if False:
        return 10
    "\n    Run one or more named munin plugins\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' munin.run uptime\n        salt '*' munin.run uptime,cpu,load,memory\n    "
    all_plugins = list_plugins()
    if isinstance(plugins, str):
        plugins = plugins.split(',')
    data = {}
    for plugin in plugins:
        if plugin not in all_plugins:
            continue
        data[plugin] = {}
        muninout = __salt__['cmd.run']('munin-run {}'.format(plugin), python_shell=False)
        for line in muninout.split('\n'):
            if 'value' in line:
                (key, val) = line.split(' ')
                key = key.split('.')[0]
                try:
                    if '.' in val:
                        val = float(val)
                    else:
                        val = int(val)
                    data[plugin][key] = val
                except ValueError:
                    pass
    return data

def run_all():
    if False:
        while True:
            i = 10
    "\n    Run all the munin plugins\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' munin.run_all\n    "
    plugins = list_plugins()
    ret = {}
    for plugin in plugins:
        ret.update(run(plugin))
    return ret

def list_plugins():
    if False:
        for i in range(10):
            print('nop')
    "\n    List all the munin plugins\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' munin.list_plugins\n    "
    pluginlist = os.listdir(PLUGINDIR)
    ret = []
    for plugin in pluginlist:
        statf = os.path.join(PLUGINDIR, plugin)
        try:
            executebit = stat.S_IXUSR & os.stat(statf)[stat.ST_MODE]
        except OSError:
            pass
        if executebit:
            ret.append(plugin)
    return ret