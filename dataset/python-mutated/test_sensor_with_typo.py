from __future__ import absolute_import
from st2reactor.sensor.base import Sensor
typobar

class TestSensorWithTypo(Sensor):

    def setup(self):
        if False:
            print('Hello World!')
        pass

    def run(self):
        if False:
            while True:
                i = 10
        pass

    def cleanup(self):
        if False:
            print('Hello World!')
        pass

    def add_trigger(self, trigger):
        if False:
            while True:
                i = 10
        pass

    def update_trigger(self, trigger):
        if False:
            while True:
                i = 10
        pass

    def remove_trigger(self, trigger):
        if False:
            for i in range(10):
                print('nop')
        pass