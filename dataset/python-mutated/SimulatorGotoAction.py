import xml.etree.ElementTree as ET
from urh.simulator.SimulatorCounterAction import SimulatorCounterAction
from urh.simulator.SimulatorItem import SimulatorItem
from urh.simulator.SimulatorProtocolLabel import SimulatorProtocolLabel
from urh.simulator.SimulatorRule import SimulatorRule, SimulatorRuleCondition, ConditionType
from urh.simulator.SimulatorTriggerCommandAction import SimulatorTriggerCommandAction

class SimulatorGotoAction(SimulatorItem):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.goto_target = None

    def set_parent(self, value):
        if False:
            return 10
        if value is not None:
            assert value.parent() is None or isinstance(value, SimulatorRuleCondition)
        super().set_parent(value)

    @property
    def target(self):
        if False:
            print('Hello World!')
        return self.simulator_config.item_dict[self.goto_target] if self.validate() else None

    def validate(self):
        if False:
            while True:
                i = 10
        target = self.simulator_config.item_dict.get(self.goto_target, None)
        return self.is_valid_goto_target(self.goto_target, target)

    def get_valid_goto_targets(self):
        if False:
            while True:
                i = 10
        valid_targets = []
        for (key, value) in self.simulator_config.item_dict.items():
            if value != self and SimulatorGotoAction.is_valid_goto_target(key, value):
                valid_targets.append(key)
        return valid_targets

    def to_xml(self) -> ET.Element:
        if False:
            while True:
                i = 10
        attributes = dict()
        if self.goto_target is not None:
            attributes['goto_target'] = self.goto_target
        return ET.Element('simulator_goto_action', attrib=attributes)

    @classmethod
    def from_xml(cls, tag: ET.Element):
        if False:
            print('Hello World!')
        result = SimulatorGotoAction()
        result.goto_target = tag.get('goto_target', None)
        return result

    @staticmethod
    def is_valid_goto_target(caption: str, item: SimulatorItem):
        if False:
            while True:
                i = 10
        if item is None:
            return False
        if isinstance(item, SimulatorProtocolLabel) or isinstance(item, SimulatorRule):
            return False
        if isinstance(item, SimulatorRuleCondition) and item.type != ConditionType.IF:
            return False
        if isinstance(item, SimulatorCounterAction):
            return False
        if isinstance(item, SimulatorTriggerCommandAction) and caption.endswith('rc'):
            return False
        return True