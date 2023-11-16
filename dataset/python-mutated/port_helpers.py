from __future__ import annotations
import socket
HOST = '127.0.0.1'
HOSTv6 = '::1'

def find_unused_port(family: socket.AddressFamily=socket.AF_INET, socktype: socket.SocketKind=socket.SOCK_STREAM) -> int:
    if False:
        print('Hello World!')
    "Returns an unused port that should be suitable for binding.  This is\n    achieved by creating a temporary socket with the same family and type as\n    the 'sock' parameter (default is AF_INET, SOCK_STREAM), and binding it to\n    the specified host address (defaults to 0.0.0.0) with the port set to 0,\n    eliciting an unused ephemeral port from the OS.  The temporary socket is\n    then closed and deleted, and the ephemeral port is returned.\n\n    Either this method or bind_port() should be used for any tests where a\n    server socket needs to be bound to a particular port for the duration of\n    the test.  Which one to use depends on whether the calling code is creating\n    a python socket, or if an unused port needs to be provided in a constructor\n    or passed to an external program (i.e. the -accept argument to openssl's\n    s_server mode).  Always prefer bind_port() over find_unused_port() where\n    possible.  Hard coded ports should *NEVER* be used.  As soon as a server\n    socket is bound to a hard coded port, the ability to run multiple instances\n    of the test simultaneously on the same host is compromised, which makes the\n    test a ticking time bomb in a buildbot environment. On Unix buildbots, this\n    may simply manifest as a failed test, which can be recovered from without\n    intervention in most cases, but on Windows, the entire python process can\n    completely and utterly wedge, requiring someone to log in to the buildbot\n    and manually kill the affected process.\n\n    (This is easy to reproduce on Windows, unfortunately, and can be traced to\n    the SO_REUSEADDR socket option having different semantics on Windows versus\n    Unix/Linux.  On Unix, you can't have two AF_INET SOCK_STREAM sockets bind,\n    listen and then accept connections on identical host/ports.  An EADDRINUSE\n    OSError will be raised at some point (depending on the platform and\n    the order bind and listen were called on each socket).\n\n    However, on Windows, if SO_REUSEADDR is set on the sockets, no EADDRINUSE\n    will ever be raised when attempting to bind two identical host/ports. When\n    accept() is called on each socket, the second caller's process will steal\n    the port from the first caller, leaving them both in an awkwardly wedged\n    state where they'll no longer respond to any signals or graceful kills, and\n    must be forcibly killed via OpenProcess()/TerminateProcess().\n\n    The solution on Windows is to use the SO_EXCLUSIVEADDRUSE socket option\n    instead of SO_REUSEADDR, which effectively affords the same semantics as\n    SO_REUSEADDR on Unix.  Given the propensity of Unix developers in the Open\n    Source world compared to Windows ones, this is a common mistake.  A quick\n    look over OpenSSL's 0.9.8g source shows that they use SO_REUSEADDR when\n    openssl.exe is called with the 's_server' option, for example. See\n    http://bugs.python.org/issue2550 for more info.  The following site also\n    has a very thorough description about the implications of both REUSEADDR\n    and EXCLUSIVEADDRUSE on Windows:\n    http://msdn2.microsoft.com/en-us/library/ms740621(VS.85).aspx)\n\n    XXX: although this approach is a vast improvement on previous attempts to\n    elicit unused ports, it rests heavily on the assumption that the ephemeral\n    port returned to us by the OS won't immediately be dished back out to some\n    other process when we close and delete our temporary socket but before our\n    calling code has a chance to bind the returned port.  We can deal with this\n    issue if/when we come across it."
    with socket.socket(family, socktype) as tempsock:
        port = bind_port(tempsock)
    del tempsock
    return port

def bind_port(sock: socket.socket, host: str=HOST) -> int:
    if False:
        for i in range(10):
            print('nop')
    "Bind the socket to a free port and return the port number. Relies on\n    ephemeral ports in order to ensure we are using an unbound port.  This is\n    important as many tests may be running simultaneously, especially in a\n    buildbot environment.  This method raises an exception if the sock.family\n    is AF_INET and sock.type is SOCK_STREAM, *and* the socket has SO_REUSEADDR\n    or SO_REUSEPORT set on it.  Tests should *never* set these socket options\n    for TCP/IP sockets.  The only case for setting these options is testing\n    multicasting via multiple UDP sockets.\n\n    Additionally, if the SO_EXCLUSIVEADDRUSE socket option is available (i.e.\n    on Windows), it will be set on the socket.  This will prevent anyone else\n    from bind()'ing to our host/port for the duration of the test.\n    "
    if sock.family == socket.AF_INET and sock.type == socket.SOCK_STREAM:
        if hasattr(socket, 'SO_REUSEADDR'):
            if sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR) == 1:
                raise ValueError('tests should never set the SO_REUSEADDR socket option on TCP/IP sockets!')
        if hasattr(socket, 'SO_REUSEPORT'):
            try:
                if sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT) == 1:
                    raise ValueError('tests should never set the SO_REUSEPORT socket option on TCP/IP sockets!')
            except OSError:
                pass
        if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    sock.bind((host, 0))
    port = sock.getsockname()[1]
    assert isinstance(port, int)
    return port