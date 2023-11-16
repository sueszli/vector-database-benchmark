"""
Control a salt cloud system
"""
import salt.utils.data
import salt.utils.json
HAS_CLOUD = False
try:
    import saltcloud
    HAS_CLOUD = True
except ImportError:
    pass
__virtualname__ = 'saltcloud'

def __virtual__():
    if False:
        return 10
    '\n    Only load if salt cloud is installed\n    '
    if HAS_CLOUD:
        return __virtualname__
    return (False, 'The saltcloudmod execution module failed to load: requires the saltcloud library.')

def create(name, profile):
    if False:
        while True:
            i = 10
    '\n    Create the named vm\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt <minion-id> saltcloud.create webserver rackspace_centos_512\n    '
    cmd = 'salt-cloud --out json -p {} {}'.format(profile, name)
    out = __salt__['cmd.run_stdout'](cmd, python_shell=False)
    try:
        ret = salt.utils.json.loads(out)
    except ValueError:
        ret = {}
    return ret