import os
from st2reactor.sensor.base import PollingSensor

class FibonacciSensor(PollingSensor):

    def __init__(self, sensor_service, config, poll_interval=20):
        if False:
            return 10
        super(FibonacciSensor, self).__init__(sensor_service=sensor_service, config=config, poll_interval=poll_interval)
        self.a = None
        self.b = None
        self.count = None
        self.logger = None

    def setup(self):
        if False:
            return 10
        self.a = 0
        self.b = 1
        self.count = 2
        self.logger = self.sensor_service.get_logger(name=self.__class__.__name__)

    def poll(self):
        if False:
            for i in range(10):
                print('nop')
        if self.a > 10000 or self.b > 10000:
            self.logger.debug('Reseting values to avoid integer overflow issues')
            self.a = 0
            self.b = 1
            self.count = 2
        fib = self.a + self.b
        self.logger.debug('Count: %d, a: %d, b: %d, fib: %s', self.count, self.a, self.b, fib)
        payload = {'count': self.count, 'fibonacci': fib, 'pythonpath': os.environ.get('PYTHONPATH', None)}
        self.sensor_service.dispatch(trigger='examples.fibonacci', payload=payload)
        self.a = self.b
        self.b = fib
        self.count = self.count + 1

    def cleanup(self):
        if False:
            while True:
                i = 10
        pass

    def add_trigger(self, trigger):
        if False:
            i = 10
            return i + 15
        pass

    def update_trigger(self, trigger):
        if False:
            while True:
                i = 10
        pass

    def remove_trigger(self, trigger):
        if False:
            print('Hello World!')
        pass