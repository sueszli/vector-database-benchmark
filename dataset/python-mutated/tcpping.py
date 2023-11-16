import socket
import time
from timeit import default_timer as timer
from six.moves import zip_longest

def avg(x):
    if False:
        i = 10
        return i + 15
    return sum(x) / float(len(x))

class Socket(object):

    def __init__(self, family, type_, timeout):
        if False:
            while True:
                i = 10
        s = socket.socket(family, type_)
        s.settimeout(timeout)
        self._s = s

    def connect(self, host, port=80):
        if False:
            for i in range(10):
                print('nop')
        self._s.connect((host, int(port)))

    def shutdown(self):
        if False:
            while True:
                i = 10
        self._s.shutdown(socket.SHUT_RD)

    def close(self):
        if False:
            return 10
        self._s.close()

class Timer(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._start = 0
        self._stop = 0

    def start(self):
        if False:
            print('Hello World!')
        self._start = timer()

    def stop(self):
        if False:
            while True:
                i = 10
        self._stop = timer()

    def cost(self, funcs, args):
        if False:
            for i in range(10):
                print('nop')
        self.start()
        for (func, arg) in zip_longest(funcs, args):
            if arg:
                func(*arg)
            else:
                func()
        self.stop()
        return self._stop - self._start

class Ping(object):

    def __init__(self, host: str, port: int, timeout=1):
        if False:
            return 10
        self.timer = Timer()
        self._successes = 0
        self._failed = 0
        self._conn_times = []
        self._host = host
        self._port = port
        self._timeout = timeout

    def _create_socket(self, family, type_):
        if False:
            for i in range(10):
                print('nop')
        return Socket(family, type_, self._timeout)

    def _success_rate(self):
        if False:
            print('Hello World!')
        count = self._successes + self._failed
        try:
            rate = float(self._successes) / count
            rate = '{0:.2f}'.format(rate)
        except ZeroDivisionError:
            rate = '0.00'
        return float(rate)

    def _get_conn_times(self) -> [float]:
        if False:
            for i in range(10):
                print('nop')
        return self._conn_times if self._conn_times != [] else [0]

    def get_maximum(self) -> float:
        if False:
            while True:
                i = 10
        return max(self._get_conn_times())

    def get_minimum(self) -> float:
        if False:
            i = 10
            return i + 15
        return min(self._get_conn_times())

    def get_average(self) -> float:
        if False:
            return 10
        return avg(self._get_conn_times())

    def get_success_rate(self):
        if False:
            print('Hello World!')
        return self._success_rate()

    def ping(self, count=10, sleep=0.3):
        if False:
            print('Hello World!')
        for n in range(1, count + 1):
            s = self._create_socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                time.sleep(sleep)
                cost_time = self.timer.cost((s.connect, s.shutdown), ((self._host, self._port), None))
                s_runtime = 1000 * cost_time
                self._conn_times.append(s_runtime)
            except socket.timeout:
                self._failed += 1
            except ConnectionResetError:
                self._failed += 1
            else:
                self._successes += 1
            finally:
                s.close()

def ping(host: str, port: int, count: int=10, sleep: float=0.2) -> (int, float):
    if False:
        while True:
            i = 10
    '\n    Ping a server and port with tcp socket\n    :param host: the hostname\n    :param port: the port number\n    :param count: number of connection tries, by default it is 10\n    :param sleep: length of sleep time in between sequent pings, by default it is 0.3\n    :return: a tuple for (average_latency, success_rate)\n    '
    p = Ping(host=host, port=port)
    p.ping(count=count, sleep=sleep)
    return (p.get_average(), p.get_success_rate())