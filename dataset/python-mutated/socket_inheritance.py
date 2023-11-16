"""
Created on 29 Nov 2013

@author: charles

Code taken from https://mail.python.org/pipermail/python-dev/2007-June/073745.html
modified to make it work
"""

def get_socket_inherit(s):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns True if the socket has been set to allow inheritance across\n    forks and execs to child processes, otherwise False\n    '
    try:
        return s.get_inheritable()
    except Exception:
        import traceback
        traceback.print_exc()

def set_socket_inherit(s, inherit=False):
    if False:
        i = 10
        return i + 15
    '\n    Mark a socket as inheritable or non-inheritable to child processes.\n\n    This should be called right after socket creation if you want\n    to prevent the socket from being inherited by child processes.\n\n    Note that for sockets, a new socket returned from accept() will be\n    inheritable even if the listener socket was not; so you should call\n    set_socket_inherit for the new socket as well.\n    '
    try:
        s.set_inheritable(inherit)
    except Exception:
        import traceback
        traceback.print_exc()

def test():
    if False:
        print('Hello World!')
    import socket
    s = socket.socket()
    orig = get_socket_inherit(s)
    set_socket_inherit(s, orig ^ True)
    if orig == get_socket_inherit(s):
        raise RuntimeError('Failed to change socket inheritance status')
    print('OK!')
if __name__ == '__main__':
    test()