from __future__ import annotations
import sys
from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING, Any
import trio
from trio._core._multierror import MultiError
from trio.socket import SOCK_STREAM, SocketType, getaddrinfo, socket
if TYPE_CHECKING:
    from collections.abc import Generator
    from socket import AddressFamily, SocketKind
if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup
DEFAULT_DELAY = 0.25

@contextmanager
def close_all() -> Generator[set[SocketType], None, None]:
    if False:
        while True:
            i = 10
    sockets_to_close: set[SocketType] = set()
    try:
        yield sockets_to_close
    finally:
        errs = []
        for sock in sockets_to_close:
            try:
                sock.close()
            except BaseException as exc:
                errs.append(exc)
        if len(errs) == 1:
            raise errs[0]
        elif errs:
            raise MultiError(errs)

def reorder_for_rfc_6555_section_5_4(targets: list[tuple[AddressFamily, SocketKind, int, str, Any]]) -> None:
    if False:
        while True:
            i = 10
    for i in range(1, len(targets)):
        if targets[i][0] != targets[0][0]:
            if i != 1:
                targets.insert(1, targets.pop(i))
            break

def format_host_port(host: str | bytes, port: int | str) -> str:
    if False:
        print('Hello World!')
    host = host.decode('ascii') if isinstance(host, bytes) else host
    if ':' in host:
        return f'[{host}]:{port}'
    else:
        return f'{host}:{port}'

async def open_tcp_stream(host: str | bytes, port: int, *, happy_eyeballs_delay: float | None=DEFAULT_DELAY, local_address: str | None=None) -> trio.SocketStream:
    """Connect to the given host and port over TCP.

    If the given ``host`` has multiple IP addresses associated with it, then
    we have a problem: which one do we use?

    One approach would be to attempt to connect to the first one, and then if
    that fails, attempt to connect to the second one ... until we've tried all
    of them. But the problem with this is that if the first IP address is
    unreachable (for example, because it's an IPv6 address and our network
    discards IPv6 packets), then we might end up waiting tens of seconds for
    the first connection attempt to timeout before we try the second address.

    Another approach would be to attempt to connect to all of the addresses at
    the same time, in parallel, and then use whichever connection succeeds
    first, abandoning the others. This would be fast, but create a lot of
    unnecessary load on the network and the remote server.

    This function strikes a balance between these two extremes: it works its
    way through the available addresses one at a time, like the first
    approach; but, if ``happy_eyeballs_delay`` seconds have passed and it's
    still waiting for an attempt to succeed or fail, then it gets impatient
    and starts the next connection attempt in parallel. As soon as any one
    connection attempt succeeds, all the other attempts are cancelled. This
    avoids unnecessary load because most connections will succeed after just
    one or two attempts, but if one of the addresses is unreachable then it
    doesn't slow us down too much.

    This is known as a "happy eyeballs" algorithm, and our particular variant
    is modelled after how Chrome connects to webservers; see `RFC 6555
    <https://tools.ietf.org/html/rfc6555>`__ for more details.

    Args:
      host (str or bytes): The host to connect to. Can be an IPv4 address,
          IPv6 address, or a hostname.

      port (int): The port to connect to.

      happy_eyeballs_delay (float or None): How many seconds to wait for each
          connection attempt to succeed or fail before getting impatient and
          starting another one in parallel. Set to `None` if you want
          to limit to only one connection attempt at a time (like
          :func:`socket.create_connection`). Default: 0.25 (250 ms).

      local_address (None or str): The local IP address or hostname to use as
          the source for outgoing connections. If ``None``, we let the OS pick
          the source IP.

          This is useful in some exotic networking configurations where your
          host has multiple IP addresses, and you want to force the use of a
          specific one.

          Note that if you pass an IPv4 ``local_address``, then you won't be
          able to connect to IPv6 hosts, and vice-versa. If you want to take
          advantage of this to force the use of IPv4 or IPv6 without
          specifying an exact source address, you can use the IPv4 wildcard
          address ``local_address="0.0.0.0"``, or the IPv6 wildcard address
          ``local_address="::"``.

    Returns:
      SocketStream: a :class:`~trio.abc.Stream` connected to the given server.

    Raises:
      OSError: if the connection fails.

    See also:
      open_ssl_over_tcp_stream

    """
    if not isinstance(host, (str, bytes)):
        raise ValueError(f'host must be str or bytes, not {host!r}')
    if not isinstance(port, int):
        raise TypeError(f'port must be int, not {port!r}')
    if happy_eyeballs_delay is None:
        happy_eyeballs_delay = DEFAULT_DELAY
    targets = await getaddrinfo(host, port, type=SOCK_STREAM)
    if not targets:
        msg = f'no results found for hostname lookup: {format_host_port(host, port)}'
        raise OSError(msg)
    reorder_for_rfc_6555_section_5_4(targets)
    oserrors: list[OSError] = []
    winning_socket: SocketType | None = None

    async def attempt_connect(socket_args: tuple[AddressFamily, SocketKind, int], sockaddr: Any, attempt_failed: trio.Event) -> None:
        nonlocal winning_socket
        try:
            sock = socket(*socket_args)
            open_sockets.add(sock)
            if local_address is not None:
                with suppress(OSError, AttributeError):
                    sock.setsockopt(trio.socket.IPPROTO_IP, trio.socket.IP_BIND_ADDRESS_NO_PORT, 1)
                try:
                    await sock.bind((local_address, 0))
                except OSError:
                    raise OSError(f'local_address={local_address!r} is incompatible with remote address {sockaddr!r}') from None
            await sock.connect(sockaddr)
            winning_socket = sock
            nursery.cancel_scope.cancel()
        except OSError as exc:
            oserrors.append(exc)
            attempt_failed.set()
    with close_all() as open_sockets:
        async with trio.open_nursery() as nursery:
            for (address_family, socket_type, proto, _, addr) in targets:
                attempt_failed = trio.Event()
                if TYPE_CHECKING:
                    await attempt_connect((address_family, socket_type, proto), addr, attempt_failed)
                nursery.start_soon(attempt_connect, (address_family, socket_type, proto), addr, attempt_failed)
                with trio.move_on_after(happy_eyeballs_delay):
                    await attempt_failed.wait()
        if winning_socket is None:
            assert len(oserrors) == len(targets)
            msg = f'all attempts to connect to {format_host_port(host, port)} failed'
            raise OSError(msg) from ExceptionGroup(msg, oserrors)
        else:
            stream = trio.SocketStream(winning_socket)
            open_sockets.remove(winning_socket)
            return stream