def patch():
    if False:
        return 10
    '\n    Patch startService and stopService so that they check the previous state\n    first.\n\n    (used for debugging only)\n    '
    from twisted.application.service import Service
    old_startService = Service.startService
    old_stopService = Service.stopService

    def startService(self):
        if False:
            while True:
                i = 10
        assert not self.running, f'{repr(self)} already running'
        return old_startService(self)

    def stopService(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.running, f'{repr(self)} already stopped'
        return old_stopService(self)
    Service.startService = startService
    Service.stopService = stopService