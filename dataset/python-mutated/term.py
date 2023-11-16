"""
Terminal creation and cleanup.
Utility functions to run a terminal (connected via socat(1)) on each host.

Requires socat(1) and xterm(1).
Optionally uses gnome-terminal.
"""
from os import environ
from mininet.log import error
from mininet.util import quietRun, errRun

def tunnelX11(node, display=None):
    if False:
        return 10
    'Create an X11 tunnel from node:6000 to the root host\n       display: display on root host (optional)\n       returns: node $DISPLAY, Popen object for tunnel'
    if display is None and 'DISPLAY' in environ:
        display = environ['DISPLAY']
    if display is None:
        error('Error: Cannot connect to display\n')
        return (None, None)
    (host, screen) = display.split(':')
    if not host or host == 'unix':
        quietRun('xhost +si:localuser:root')
        return (display, None)
    else:
        port = 6000 + int(float(screen))
        connection = 'TCP\\:%s\\:%s' % (host, port)
        cmd = ['socat', 'TCP-LISTEN:%d,fork,reuseaddr' % port, "EXEC:'mnexec -a 1 socat STDIO %s'" % connection]
    return ('localhost:' + screen, node.popen(cmd))

def makeTerm(node, title='Node', term='xterm', display=None, cmd='bash'):
    if False:
        while True:
            i = 10
    "Create an X11 tunnel to the node and start up a terminal.\n       node: Node object\n       title: base title\n       term: 'xterm' or 'gterm'\n       returns: two Popen objects, tunnel and terminal"
    title = '"%s: %s"' % (title, node.name)
    if not node.inNamespace:
        title += ' (root)'
    cmds = {'xterm': ['xterm', '-title', title, '-display'], 'gterm': ['gnome-terminal', '--title', title, '--display']}
    if term not in cmds:
        error('invalid terminal type: %s' % term)
        return None
    (display, tunnel) = tunnelX11(node, display)
    if display is None:
        return []
    term = node.popen(cmds[term] + [display, '-e', 'env TERM=ansi %s' % cmd])
    return [tunnel, term] if tunnel else [term]

def runX11(node, cmd):
    if False:
        while True:
            i = 10
    'Run an X11 client on a node'
    (_display, tunnel) = tunnelX11(node)
    if _display is None:
        return []
    popen = node.popen(cmd)
    return [tunnel, popen]

def cleanUpScreens():
    if False:
        print('Hello World!')
    'Remove moldy socat X11 tunnels.'
    errRun('pkill -9 -f mnexec.*socat')

def makeTerms(nodes, title='Node', term='xterm'):
    if False:
        return 10
    'Create terminals.\n       nodes: list of Node objects\n       title: base title for each\n       returns: list of created tunnel/terminal processes'
    terms = []
    for node in nodes:
        terms += makeTerm(node, title, term)
    return terms