import time
from xmlrpc.client import ServerProxy, Fault
from locust import User, task

class XmlRpcClient(ServerProxy):
    """
    XmlRpcClient is a wrapper around the standard library's ServerProxy.
    It proxies any function calls and fires the *request* event when they finish,
    so that the calls get recorded in Locust.
    """

    def __init__(self, host, request_event):
        if False:
            while True:
                i = 10
        super().__init__(host)
        self._request_event = request_event

    def __getattr__(self, name):
        if False:
            print('Hello World!')
        func = ServerProxy.__getattr__(self, name)

        def wrapper(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            request_meta = {'request_type': 'xmlrpc', 'name': name, 'start_time': time.time(), 'response_length': 0, 'response': None, 'context': {}, 'exception': None}
            start_perf_counter = time.perf_counter()
            try:
                request_meta['response'] = func(*args, **kwargs)
            except Fault as e:
                request_meta['exception'] = e
            request_meta['response_time'] = (time.perf_counter() - start_perf_counter) * 1000
            self._request_event.fire(**request_meta)
            return request_meta['response']
        return wrapper

class XmlRpcUser(User):
    """
    A minimal Locust user class that provides an XmlRpcClient to its subclasses
    """
    abstract = True

    def __init__(self, environment):
        if False:
            i = 10
            return i + 15
        super().__init__(environment)
        self.client = XmlRpcClient(self.host, request_event=environment.events.request)

class MyUser(XmlRpcUser):
    host = 'http://127.0.0.1:8877/'

    @task
    def get_time(self):
        if False:
            print('Hello World!')
        self.client.get_time()

    @task
    def get_random_number(self):
        if False:
            for i in range(10):
                print('nop')
        self.client.get_random_number(0, 100)