import pytest
import unittest
from modules.sfp_iban import sfp_iban
from sflib import SpiderFoot
from spiderfoot import SpiderFootEvent, SpiderFootTarget

@pytest.mark.usefixtures
class TestModuleIban(unittest.TestCase):

    def test_opts(self):
        if False:
            i = 10
            return i + 15
        module = sfp_iban()
        self.assertEqual(len(module.opts), len(module.optdescs))

    def test_setup(self):
        if False:
            while True:
                i = 10
        sf = SpiderFoot(self.default_options)
        module = sfp_iban()
        module.setup(sf, dict())

    def test_watchedEvents_should_return_list(self):
        if False:
            i = 10
            return i + 15
        module = sfp_iban()
        self.assertIsInstance(module.watchedEvents(), list)

    def test_producedEvents_should_return_list(self):
        if False:
            i = 10
            return i + 15
        module = sfp_iban()
        self.assertIsInstance(module.producedEvents(), list)

    def test_handleEvent_event_data_containing_iban_string_should_return_event(self):
        if False:
            return 10
        sf = SpiderFoot(self.default_options)
        module = sfp_iban()
        module.setup(sf, dict())
        target_value = 'spiderfoot.net'
        target_type = 'INTERNET_NAME'
        target = SpiderFootTarget(target_value, target_type)
        module.setTarget(target)

        def new_notifyListeners(self, event):
            if False:
                print('Hello World!')
            expected = 'IBAN_NUMBER'
            if str(event.eventType) != expected:
                raise Exception(f'{event.eventType} != {expected}')
            expected = 'BE71096123456769'
            if str(event.data) != expected:
                raise Exception(f'{event.data} != {expected}')
            raise Exception('OK')
        module.notifyListeners = new_notifyListeners.__get__(module, sfp_iban)
        event_type = 'ROOT'
        event_data = 'BE71096123456769'
        event_module = ''
        source_event = ''
        evt = SpiderFootEvent(event_type, event_data, event_module, source_event)
        with self.assertRaises(Exception) as cm:
            module.handleEvent(evt)
        self.assertEqual('OK', str(cm.exception))

    def test_handleEvent_event_data_not_containing_iban_string_should_not_return_event(self):
        if False:
            for i in range(10):
                print('nop')
        sf = SpiderFoot(self.default_options)
        module = sfp_iban()
        module.setup(sf, dict())
        target_value = 'spiderfoot.net'
        target_type = 'INTERNET_NAME'
        target = SpiderFootTarget(target_value, target_type)
        module.setTarget(target)

        def new_notifyListeners(self, event):
            if False:
                for i in range(10):
                    print('nop')
            raise Exception(f'Raised event {event.eventType}: {event.data}')
        module.notifyListeners = new_notifyListeners.__get__(module, sfp_iban)
        event_type = 'ROOT'
        event_data = 'example data'
        event_module = ''
        source_event = ''
        evt = SpiderFootEvent(event_type, event_data, event_module, source_event)
        result = module.handleEvent(evt)
        self.assertIsNone(result)