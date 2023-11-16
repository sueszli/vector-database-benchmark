import xml.etree.ElementTree as ET
from urh.simulator.SimulatorItem import SimulatorItem
from urh.simulator.SimulatorRule import SimulatorRuleCondition
from urh.util.Formatter import Formatter

class SimulatorCounterAction(SimulatorItem):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.start = 1
        self.step = 1
        self.__value = self.start

    @property
    def value(self):
        if False:
            print('Hello World!')
        return self.__value

    def reset_value(self):
        if False:
            return 10
        self.__value = self.start

    def progress_value(self):
        if False:
            print('Hello World!')
        self.__value += self.step

    def validate(self):
        if False:
            return 10
        return True

    def set_parent(self, value):
        if False:
            while True:
                i = 10
        if value is not None:
            assert value.parent() is None or isinstance(value, SimulatorRuleCondition)
        super().set_parent(value)

    def to_xml(self):
        if False:
            print('Hello World!')
        attrib = {'start': str(self.start), 'step': str(self.step)}
        return ET.Element('simulator_counter_action', attrib=attrib)

    @classmethod
    def from_xml(cls, tag):
        if False:
            for i in range(10):
                print('nop')
        result = SimulatorCounterAction()
        result.start = Formatter.str2val(tag.get('start', '1'), int, 1)
        result.step = Formatter.str2val(tag.get('step', '1'), int, 1)
        return result