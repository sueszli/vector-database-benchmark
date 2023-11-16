import pytest
import unittest
from modules.sfp_phone import sfp_phone
from sflib import SpiderFoot
from spiderfoot import SpiderFootEvent, SpiderFootTarget

@pytest.mark.usefixtures
class TestModulePhone(unittest.TestCase):

    def test_opts(self):
        if False:
            i = 10
            return i + 15
        module = sfp_phone()
        self.assertEqual(len(module.opts), len(module.optdescs))

    def test_setup(self):
        if False:
            return 10
        sf = SpiderFoot(self.default_options)
        module = sfp_phone()
        module.setup(sf, dict())

    def test_watchedEvents_should_return_list(self):
        if False:
            print('Hello World!')
        module = sfp_phone()
        self.assertIsInstance(module.watchedEvents(), list)

    def test_producedEvents_should_return_list(self):
        if False:
            while True:
                i = 10
        module = sfp_phone()
        self.assertIsInstance(module.producedEvents(), list)

    def test_handleEvent_domain_whois_event_data_containing_phone_string_should_create_phone_number_event(self):
        if False:
            i = 10
            return i + 15
        sf = SpiderFoot(self.default_options)
        module = sfp_phone()
        module.setup(sf, dict())
        target_value = 'spiderfoot.net'
        target_type = 'INTERNET_NAME'
        target = SpiderFootTarget(target_value, target_type)
        module.setTarget(target)

        def new_notifyListeners(self, event):
            if False:
                print('Hello World!')
            expected = 'PHONE_NUMBER'
            if str(event.eventType) != expected:
                raise Exception(f'{event.eventType} != {expected}')
            expected = '+12025550111'
            if str(event.data) != expected:
                raise Exception(f'{event.data} != {expected}')
            raise Exception('OK')
        module.notifyListeners = new_notifyListeners.__get__(module, sfp_phone)
        event_type = 'ROOT'
        event_data = 'example data'
        event_module = ''
        source_event = ''
        evt = SpiderFootEvent(event_type, event_data, event_module, source_event)
        event_type = 'DOMAIN_WHOIS'
        event_data = 'example data +1 202 555 0111 example data'
        event_module = 'example module'
        source_event = evt
        evt = SpiderFootEvent(event_type, event_data, event_module, source_event)
        with self.assertRaises(Exception) as cm:
            module.handleEvent(evt)
        self.assertEqual('OK', str(cm.exception))

    def test_handleEvent_domain_whois_event_data_not_containing_phone_string_should_not_create_event(self):
        if False:
            return 10
        sf = SpiderFoot(self.default_options)
        module = sfp_phone()
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
        module.notifyListeners = new_notifyListeners.__get__(module, sfp_phone)
        event_type = 'ROOT'
        event_data = 'example data'
        event_module = ''
        source_event = ''
        evt = SpiderFootEvent(event_type, event_data, event_module, source_event)
        event_type = 'DOMAIN_WHOIS'
        event_data = 'example data'
        event_module = 'example module'
        source_event = evt
        evt = SpiderFootEvent(event_type, event_data, event_module, source_event)
        result = module.handleEvent(evt)
        self.assertIsNone(result)

    def test_handleEvent_phone_number_event_data_containing_phone_string_should_return_provider_telco_event(self):
        if False:
            for i in range(10):
                print('nop')
        sf = SpiderFoot(self.default_options)
        module = sfp_phone()
        module.setup(sf, dict())
        target_value = 'spiderfoot.net'
        target_type = 'INTERNET_NAME'
        target = SpiderFootTarget(target_value, target_type)
        module.setTarget(target)

        def new_notifyListeners(self, event):
            if False:
                print('Hello World!')
            expected = 'PROVIDER_TELCO'
            if str(event.eventType) != expected:
                raise Exception(f'{event.eventType} != {expected}')
            expected = 'Swisscom'
            if str(event.data) != expected:
                raise Exception(f'{event.data} != {expected}')
            raise Exception('OK')
        module.notifyListeners = new_notifyListeners.__get__(module, sfp_phone)
        event_type = 'ROOT'
        event_data = 'example data'
        event_module = ''
        source_event = ''
        evt = SpiderFootEvent(event_type, event_data, event_module, source_event)
        event_type = 'PHONE_NUMBER'
        event_data = '+41798765432'
        event_module = 'example module'
        source_event = evt
        evt = SpiderFootEvent(event_type, event_data, event_module, source_event)
        with self.assertRaises(Exception) as cm:
            module.handleEvent(evt)
        self.assertEqual('OK', str(cm.exception))