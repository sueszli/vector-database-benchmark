from twisted.logger import Logger

class MyObject:
    log = Logger()

    def __init__(self, value):
        if False:
            return 10
        self.value = value

    def doSomething(self, something):
        if False:
            return 10
        self.log.info('Object with value {log_source.value!r} doing {something}.', something=something)
MyObject(7).doSomething('a task')