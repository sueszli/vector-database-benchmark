import pytest
import unittest
from modules.sfp_errors import sfp_errors
from sflib import SpiderFoot
from spiderfoot import SpiderFootEvent, SpiderFootTarget

@pytest.mark.usefixtures
class TestModuleErrors(unittest.TestCase):

    def test_opts(self):
        if False:
            for i in range(10):
                print('nop')
        module = sfp_errors()
        self.assertEqual(len(module.opts), len(module.optdescs))

    def test_setup(self):
        if False:
            i = 10
            return i + 15
        sf = SpiderFoot(self.default_options)
        module = sfp_errors()
        module.setup(sf, dict())

    def test_watchedEvents_should_return_list(self):
        if False:
            for i in range(10):
                print('nop')
        module = sfp_errors()
        self.assertIsInstance(module.watchedEvents(), list)

    def test_producedEvents_should_return_list(self):
        if False:
            i = 10
            return i + 15
        module = sfp_errors()
        self.assertIsInstance(module.producedEvents(), list)

    def test_handleEvent_should_only_handle_events_from_sfp_spider(self):
        if False:
            for i in range(10):
                print('nop')
        sf = SpiderFoot(self.default_options)
        module = sfp_errors()
        module.setup(sf, dict())
        target_value = 'spiderfoot.net'
        target_type = 'INTERNET_NAME'
        target = SpiderFootTarget(target_value, target_type)
        module.setTarget(target)

        def new_notifyListeners(self, event):
            if False:
                i = 10
                return i + 15
            raise Exception(f'Raised event {event.eventType}: {event.data}')
        module.notifyListeners = new_notifyListeners.__get__(module, sfp_errors)
        event_type = 'ROOT'
        event_data = 'example data'
        event_module = ''
        source_event = ''
        evt = SpiderFootEvent(event_type, event_data, event_module, source_event)
        event_type = 'TARGET_WEB_CONTENT'
        event_data = 'example data Internal Server Error example data'
        event_module = 'something else entirely'
        source_event = evt
        evt = SpiderFootEvent(event_type, event_data, event_module, source_event)
        evt.actualSource = 'https://spiderfoot.net/'
        result = module.handleEvent(evt)
        self.assertIsNone(result)

    def test_handleEvent_should_only_handle_events_within_target_scope(self):
        if False:
            for i in range(10):
                print('nop')
        sf = SpiderFoot(self.default_options)
        module = sfp_errors()
        module.setup(sf, dict())
        target_value = 'spiderfoot.net'
        target_type = 'INTERNET_NAME'
        target = SpiderFootTarget(target_value, target_type)
        module.setTarget(target)

        def new_notifyListeners(self, event):
            if False:
                return 10
            raise Exception(f'Raised event {event.eventType}: {event.data}')
        module.notifyListeners = new_notifyListeners.__get__(module, sfp_errors)
        event_type = 'ROOT'
        event_data = 'example data'
        event_module = ''
        source_event = ''
        evt = SpiderFootEvent(event_type, event_data, event_module, source_event)
        event_type = 'TARGET_WEB_CONTENT'
        event_data = 'example data Internal Server Error example data'
        event_module = 'sfp_spider'
        source_event = evt
        evt = SpiderFootEvent(event_type, event_data, event_module, source_event)
        evt.actualSource = 'https://something.else.entirely/'
        result = module.handleEvent(evt)
        self.assertIsNone(result)

    def test_handleEvent_event_data_containing_error_string_should_return_event(self):
        if False:
            return 10
        sf = SpiderFoot(self.default_options)
        module = sfp_errors()
        module.setup(sf, dict())
        target_value = 'spiderfoot.net'
        target_type = 'INTERNET_NAME'
        target = SpiderFootTarget(target_value, target_type)
        module.setTarget(target)

        def new_notifyListeners(self, event):
            if False:
                print('Hello World!')
            expected = 'ERROR_MESSAGE'
            if str(event.eventType) != expected:
                raise Exception(f'{event.eventType} != {expected}')
            expected = 'Generic Error'
            if str(event.data) != expected:
                raise Exception(f'{event.data} != {expected}')
            raise Exception('OK')
        module.notifyListeners = new_notifyListeners.__get__(module, sfp_errors)
        event_type = 'ROOT'
        event_data = 'example data'
        event_module = ''
        source_event = ''
        evt = SpiderFootEvent(event_type, event_data, event_module, source_event)
        result = module.handleEvent(evt)
        self.assertIsNone(result)
        event_type = 'TARGET_WEB_CONTENT'
        event_data = 'example data Internal Server Error example data'
        event_module = 'sfp_spider'
        source_event = evt
        evt = SpiderFootEvent(event_type, event_data, event_module, source_event)
        evt.actualSource = 'https://spiderfoot.net/'
        with self.assertRaises(Exception) as cm:
            module.handleEvent(evt)
        self.assertEqual('OK', str(cm.exception))

    def test_handleEvent_event_data_not_containing_error_string_should_not_return_event(self):
        if False:
            return 10
        sf = SpiderFoot(self.default_options)
        module = sfp_errors()
        module.setup(sf, dict())
        target_value = 'spiderfoot.net'
        target_type = 'INTERNET_NAME'
        target = SpiderFootTarget(target_value, target_type)
        module.setTarget(target)

        def new_notifyListeners(self, event):
            if False:
                print('Hello World!')
            raise Exception(f'Raised event {event.eventType}: {event.data}')
        module.notifyListeners = new_notifyListeners.__get__(module, sfp_errors)
        event_type = 'ROOT'
        event_data = 'example data'
        event_module = ''
        source_event = ''
        evt = SpiderFootEvent(event_type, event_data, event_module, source_event)
        result = module.handleEvent(evt)
        event_type = 'TARGET_WEB_CONTENT'
        event_data = 'example data'
        event_module = 'sfp_spider'
        source_event = evt
        evt = SpiderFootEvent(event_type, event_data, event_module, source_event)
        evt.actualSource = 'https://spiderfoot.net/'
        result = module.handleEvent(evt)
        self.assertIsNone(result)