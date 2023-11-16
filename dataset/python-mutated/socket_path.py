import os

def get_socket_path():
    if False:
        while True:
            i = 10
    'Gets the path to the Ulauncher control unix socket.'
    rundir = os.environ.get('XDG_RUNTIME_DIR') or '/tmp'
    return os.path.join(rundir, 'ulauncher_control')