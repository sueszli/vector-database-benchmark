"""
uWSGI stats server https://uwsgi-docs.readthedocs.io/en/latest/StatsServer.html

:maintainer: Peter Baumgartner <pete@lincolnloop.com>
:maturity:   new
:platform:   all
"""
import salt.utils.json
import salt.utils.path

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Only load the module if uwsgi is installed\n    '
    cmd = 'uwsgi'
    if salt.utils.path.which(cmd):
        return cmd
    return (False, 'The uwsgi execution module failed to load: the uwsgi binary is not in the path.')

def stats(socket):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return the data from `uwsgi --connect-and-read` as a dictionary.\n\n    socket\n        The socket the uWSGI stats server is listening on\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' uwsgi.stats /var/run/mystatsserver.sock\n\n        salt '*' uwsgi.stats 127.0.0.1:5050\n    "
    cmd = ['uwsgi', '--connect-and-read', '{}'.format(socket)]
    out = __salt__['cmd.run'](cmd, python_shell=False)
    return salt.utils.json.loads(out)