from st2reactor.sensor.base import PollingSensor

class SamplePollingSensor(PollingSensor):
    """
    * self.sensor_service
        - provides utilities like
            get_logger() for writing to logs.
            dispatch() for dispatching triggers into the system.
    * self._config
        - contains configuration that was specified as
          config.yaml in the pack.
    * self._poll_interval
        - indicates the interval between two successive poll() calls.
    """

    def setup(self):
        if False:
            print('Hello World!')
        pass

    def poll(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def cleanup(self):
        if False:
            while True:
                i = 10
        pass

    def add_trigger(self, trigger):
        if False:
            return 10
        pass

    def update_trigger(self, trigger):
        if False:
            for i in range(10):
                print('nop')
        pass

    def remove_trigger(self, trigger):
        if False:
            for i in range(10):
                print('nop')
        pass