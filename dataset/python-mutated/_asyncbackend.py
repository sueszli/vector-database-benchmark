class NullContext:

    def __init__(self, enter_result=None):
        if False:
            return 10
        self.enter_result = enter_result

    def __enter__(self):
        if False:
            return 10
        return self.enter_result

    def __exit__(self, exc_type, exc_value, traceback):
        if False:
            return 10
        pass

    async def __aenter__(self):
        return self.enter_result

    async def __aexit__(self, exc_type, exc_value, traceback):
        pass

class Socket:

    async def close(self):
        pass

    async def getpeername(self):
        raise NotImplementedError

    async def getsockname(self):
        raise NotImplementedError

    async def getpeercert(self, timeout):
        raise NotImplementedError

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.close()

class DatagramSocket(Socket):

    def __init__(self, family: int):
        if False:
            print('Hello World!')
        self.family = family

    async def sendto(self, what, destination, timeout):
        raise NotImplementedError

    async def recvfrom(self, size, timeout):
        raise NotImplementedError

class StreamSocket(Socket):

    async def sendall(self, what, timeout):
        raise NotImplementedError

    async def recv(self, size, timeout):
        raise NotImplementedError

class NullTransport:

    async def connect_tcp(self, host, port, timeout, local_address):
        raise NotImplementedError

class Backend:

    def name(self):
        if False:
            i = 10
            return i + 15
        return 'unknown'

    async def make_socket(self, af, socktype, proto=0, source=None, destination=None, timeout=None, ssl_context=None, server_hostname=None):
        raise NotImplementedError

    def datagram_connection_required(self):
        if False:
            i = 10
            return i + 15
        return False

    async def sleep(self, interval):
        raise NotImplementedError

    def get_transport_class(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    async def wait_for(self, awaitable, timeout):
        raise NotImplementedError