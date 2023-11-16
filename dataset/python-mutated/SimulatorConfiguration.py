import random
import xml.etree.ElementTree as ET
from collections import OrderedDict
from PyQt5.QtCore import pyqtSignal, QObject
from urh import settings
from urh.signalprocessing.Encoding import Encoding
from urh.signalprocessing.FieldType import FieldType
from urh.signalprocessing.Modulator import Modulator
from urh.signalprocessing.Participant import Participant
from urh.signalprocessing.ProtocoLabel import ProtocolLabel
from urh.simulator.SimulatorCounterAction import SimulatorCounterAction
from urh.simulator.SimulatorGotoAction import SimulatorGotoAction
from urh.simulator.SimulatorItem import SimulatorItem
from urh.simulator.SimulatorMessage import SimulatorMessage
from urh.simulator.SimulatorProtocolLabel import SimulatorProtocolLabel
from urh.simulator.SimulatorRule import SimulatorRuleCondition, ConditionType, SimulatorRule
from urh.simulator.SimulatorSleepAction import SimulatorSleepAction
from urh.simulator.SimulatorTriggerCommandAction import SimulatorTriggerCommandAction
from urh.util.ProjectManager import ProjectManager

class SimulatorConfiguration(QObject):
    participants_changed = pyqtSignal()
    item_dict_updated = pyqtSignal()
    active_participants_updated = pyqtSignal()
    items_deleted = pyqtSignal(list)
    items_updated = pyqtSignal(list)
    items_moved = pyqtSignal(list)
    items_added = pyqtSignal(list)

    def __init__(self, project_manager: ProjectManager):
        if False:
            while True:
                i = 10
        super().__init__()
        self.rootItem = SimulatorItem()
        self.project_manager = project_manager
        self.broadcast_part = Participant('Broadcast', 'Broadcast', self.project_manager.broadcast_address_hex, id='broadcast_participant')
        self.__active_participants = None
        self.item_dict = OrderedDict()
        self.create_connects()

    def create_connects(self):
        if False:
            print('Hello World!')
        self.project_manager.project_updated.connect(self.on_project_updated)
        self.items_added.connect(self.update_item_dict)
        self.items_moved.connect(self.update_item_dict)
        self.items_updated.connect(self.update_item_dict)
        self.items_deleted.connect(self.update_item_dict)
        self.items_added.connect(self.update_active_participants)
        self.items_updated.connect(self.update_active_participants)
        self.items_deleted.connect(self.update_active_participants)

    @property
    def participants(self):
        if False:
            print('Hello World!')
        return self.project_manager.participants + [self.broadcast_part]

    @property
    def active_participants(self):
        if False:
            for i in range(10):
                print('nop')
        if self.__active_participants is None:
            self.update_active_participants()
        return self.__active_participants

    @property
    def rx_needed(self) -> bool:
        if False:
            i = 10
            return i + 15
        return any((hasattr(msg.destination, 'simulate') and msg.destination.simulate for msg in self.get_all_messages()))

    @property
    def tx_needed(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return any((hasattr(msg.source, 'simulate') and msg.source.simulate for msg in self.get_all_messages()))

    def update_item_dict(self):
        if False:
            while True:
                i = 10
        self.item_dict.clear()
        for item in self.get_all_items():
            if isinstance(item, SimulatorProtocolLabel):
                index = item.parent().index()
                suffix = '.' + item.name.replace(' ', '_')
            else:
                index = item.index()
                suffix = ''
            name = 'item' + index.replace('.', '_') + suffix
            if isinstance(item, SimulatorCounterAction):
                self.item_dict[name + '.counter_value'] = item
            else:
                self.item_dict[name] = item
                if isinstance(item, SimulatorTriggerCommandAction):
                    self.item_dict[name + '.rc'] = item
        self.item_dict_updated.emit()

    def update_valid_states(self):
        if False:
            for i in range(10):
                print('nop')
        for child in self.rootItem.children:
            self.__update_valid_states(child)

    @staticmethod
    def __update_valid_states(node: SimulatorItem):
        if False:
            return 10
        for child in node.children:
            SimulatorConfiguration.__update_valid_states(child)
        node.is_valid = node.validate()

    def protocol_valid(self):
        if False:
            while True:
                i = 10
        self.update_valid_states()
        return all((item.is_valid for item in self.get_all_items()))

    def on_project_updated(self):
        if False:
            i = 10
            return i + 15
        self.broadcast_part.address_hex = self.project_manager.broadcast_address_hex
        participants = self.participants
        for msg in self.get_all_messages():
            if msg.participant not in participants:
                msg.participant = None
            if msg.destination not in participants:
                msg.destination = None
        self.participants_changed.emit()

    def add_items(self, items, pos: int, parent_item):
        if False:
            for i in range(10):
                print('nop')
        if parent_item is None:
            parent_item = self.rootItem
        assert isinstance(parent_item, SimulatorItem)
        for item in items:
            parent_item.insert_child(pos, item)
            pos += 1
        self.items_added.emit(items)

    def delete_items(self, items):
        if False:
            return 10
        for (i, item) in enumerate(items):
            if isinstance(item, SimulatorRuleCondition) and item.type == ConditionType.IF:
                items[i] = item.parent()
            items[i].delete()
        self.items_deleted.emit(items)

    def move_items(self, items, new_pos: int, new_parent: SimulatorItem):
        if False:
            print('Hello World!')
        if new_parent is None:
            new_parent = self.rootItem
        for item in items:
            if item.parent() is new_parent and item.get_pos() < new_pos:
                new_pos -= 1
            new_parent.insert_child(new_pos, item)
            new_pos += 1
        self.items_moved.emit(items)

    def add_label(self, start: int, end: int, name: str=None, color_index: int=None, type: FieldType=None, parent_item: SimulatorMessage=None):
        if False:
            while True:
                i = 10
        assert isinstance(parent_item, SimulatorMessage)
        name = '' if not name else name
        used_colors = [p.color_index for p in parent_item.message_type]
        avail_colors = [i for (i, _) in enumerate(settings.LABEL_COLORS) if i not in used_colors]
        if color_index is None:
            if len(avail_colors) > 0:
                color_index = avail_colors[0]
            else:
                color_index = random.randint(0, len(settings.LABEL_COLORS) - 1)
        label = ProtocolLabel(name, start, end, color_index, type)
        sim_label = SimulatorProtocolLabel(label)
        self.add_items([sim_label], -1, parent_item)
        return sim_label

    def n_top_level_items(self):
        if False:
            while True:
                i = 10
        return self.rootItem.child_count()

    def update_active_participants(self):
        if False:
            while True:
                i = 10
        messages = self.get_all_messages()
        active_participants = []
        for part in self.project_manager.participants:
            if any((msg.participant == part or msg.destination == part for msg in messages)):
                active_participants.append(part)
        self.__active_participants = active_participants
        self.active_participants_updated.emit()

    def consolidate_messages(self):
        if False:
            while True:
                i = 10
        current_item = self.rootItem
        redundant_messages = []
        updated_messages = []
        while current_item is not None:
            if isinstance(current_item, SimulatorMessage):
                first_msg = current_item
                current_msg = current_item
                repeat_counter = 0
                while isinstance(current_msg.next_sibling(), SimulatorMessage) and current_item.plain_bits == current_msg.next_sibling().plain_bits:
                    repeat_counter += 1
                    current_msg = current_msg.next_sibling()
                    redundant_messages.append(current_msg)
                if repeat_counter:
                    first_msg.repeat += repeat_counter
                    updated_messages.append(first_msg)
                current_item = current_msg.next()
            else:
                current_item = current_item.next()
        self.delete_items(redundant_messages)
        self.items_updated.emit(updated_messages)

    def get_all_messages(self):
        if False:
            return 10
        '\n\n        :rtype: list[SimulatorMessage]\n        '
        return [item for item in self.get_all_items() if isinstance(item, SimulatorMessage)]

    def load_from_xml(self, xml_tag: ET.Element, message_types):
        if False:
            while True:
                i = 10
        assert xml_tag.tag == 'simulator_config'
        items = []
        modulators_tag = xml_tag.find('modulators')
        if modulators_tag:
            self.project_manager.modulators = Modulator.modulators_from_xml_tag(modulators_tag)
        participants_tag = xml_tag.find('participants')
        if participants_tag:
            for participant in Participant.read_participants_from_xml_tag(participants_tag):
                if participant not in self.project_manager.participants:
                    self.project_manager.participants.append(participant)
            self.participants_changed.emit()
        decodings_tag = xml_tag.find('decodings')
        if decodings_tag:
            self.project_manager.decodings = Encoding.read_decoders_from_xml_tag(decodings_tag)
        rx_config_tag = xml_tag.find('simulator_rx_conf')
        if rx_config_tag:
            ProjectManager.read_device_conf_dict(rx_config_tag, self.project_manager.simulator_rx_conf)
        tx_config_tag = xml_tag.find('simulator_tx_conf')
        if tx_config_tag:
            ProjectManager.read_device_conf_dict(tx_config_tag, self.project_manager.simulator_tx_conf)
        for child_tag in xml_tag.find('items'):
            items.append(self.load_item_from_xml(child_tag, message_types))
        self.add_items(items, pos=0, parent_item=None)

    def load_item_from_xml(self, xml_tag: ET.Element, message_types) -> SimulatorItem:
        if False:
            while True:
                i = 10
        if xml_tag.tag == 'simulator_message':
            item = SimulatorMessage.new_from_xml(xml_tag, self.participants, self.project_manager.decodings, message_types)
        elif xml_tag.tag == 'simulator_label':
            item = SimulatorProtocolLabel.from_xml(xml_tag, self.project_manager.field_types_by_caption)
        elif xml_tag.tag == 'simulator_trigger_command_action':
            item = SimulatorTriggerCommandAction.from_xml(xml_tag)
        elif xml_tag.tag == 'simulator_sleep_action':
            item = SimulatorSleepAction.from_xml(xml_tag)
        elif xml_tag.tag == 'simulator_counter_action':
            item = SimulatorCounterAction.from_xml(xml_tag)
        elif xml_tag.tag == 'simulator_rule':
            item = SimulatorRule.from_xml(xml_tag)
        elif xml_tag.tag == 'simulator_rule_condition':
            item = SimulatorRuleCondition.from_xml(xml_tag)
        elif xml_tag.tag == 'simulator_goto_action':
            item = SimulatorGotoAction.from_xml(xml_tag)
        elif xml_tag.tag in ('message', 'label', 'checksum_label'):
            return None
        else:
            raise ValueError('Unknown simulator item tag: {}'.format(xml_tag.tag))
        for child_tag in xml_tag:
            child = self.load_item_from_xml(child_tag, message_types)
            if child is not None:
                item.add_child(child)
        return item

    def save_to_xml(self, standalone=False) -> ET.Element:
        if False:
            print('Hello World!')
        result = ET.Element('simulator_config')
        if standalone:
            result.append(Modulator.modulators_to_xml_tag(self.project_manager.modulators))
            result.append(Encoding.decodings_to_xml_tag(self.project_manager.decodings))
            result.append(Participant.participants_to_xml_tag(self.project_manager.participants))
            result.append(self.project_manager.simulator_rx_conf_to_xml())
            result.append(self.project_manager.simulator_tx_conf_to_xml())
        items_tag = ET.SubElement(result, 'items')
        for item in self.rootItem.children:
            self.__save_item_to_xml(items_tag, item)
        return result

    def __save_item_to_xml(self, tag: ET.Element, item):
        if False:
            print('Hello World!')
        if isinstance(item, SimulatorMessage):
            child_tag = item.to_xml(decoders=self.project_manager.decodings, include_message_type=True, write_bits=True)
        else:
            child_tag = item.to_xml()
        tag.append(child_tag)
        for child in item.children:
            self.__save_item_to_xml(child_tag, child)

    def get_all_items(self):
        if False:
            print('Hello World!')
        items = []
        for child in self.rootItem.children:
            self.__get_all_items(child, items)
        return items

    @staticmethod
    def __get_all_items(node: SimulatorItem, items: list):
        if False:
            return 10
        items.append(node)
        for child in node.children:
            SimulatorConfiguration.__get_all_items(child, items)