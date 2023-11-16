from st2reactor.sensor.base import Sensor

class SampleSensor(Sensor):
    """
    * self.sensor_service
        - provides utilities like
            - get_logger() - returns logger instance specific to this sensor.
            - dispatch() for dispatching triggers into the system.
    * self._config
        - contains parsed configuration that was specified as
          config.yaml in the pack.
    """

    def setup(self):
        if False:
            i = 10
            return i + 15
        pass

    def run(self):
        if False:
            return 10
        pass

    def cleanup(self):
        if False:
            while True:
                i = 10
        pass

    def add_trigger(self, trigger):
        if False:
            print('Hello World!')
        pass

    def update_trigger(self, trigger):
        if False:
            while True:
                i = 10
        pass

    def remove_trigger(self, trigger):
        if False:
            return 10
        pass