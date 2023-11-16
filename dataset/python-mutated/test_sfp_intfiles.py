import pytest
import unittest
from modules.sfp_intfiles import sfp_intfiles
from sflib import SpiderFoot
from spiderfoot import SpiderFootEvent, SpiderFootTarget

@pytest.mark.usefixtures
class TestModuleIntfiles(unittest.TestCase):

    def test_opts(self):
        if False:
            for i in range(10):
                print('nop')
        module = sfp_intfiles()
        self.assertEqual(len(module.opts), len(module.optdescs))

    def test_setup(self):
        if False:
            while True:
                i = 10
        sf = SpiderFoot(self.default_options)
        module = sfp_intfiles()
        module.setup(sf, dict())

    def test_watchedEvents_should_return_list(self):
        if False:
            for i in range(10):
                print('nop')
        module = sfp_intfiles()
        self.assertIsInstance(module.watchedEvents(), list)

    def test_producedEvents_should_return_list(self):
        if False:
            i = 10
            return i + 15
        module = sfp_intfiles()
        self.assertIsInstance(module.producedEvents(), list)

    def test_handleEvent_event_data_internal_url_with_interesting_file_extension_should_return_event(self):
        if False:
            i = 10
            return i + 15
        sf = SpiderFoot(self.default_options)
        module = sfp_intfiles()
        module.setup(sf, dict())
        target_value = 'spiderfoot.net'
        target_type = 'INTERNET_NAME'
        target = SpiderFootTarget(target_value, target_type)
        module.setTarget(target)

        def new_notifyListeners(self, event):
            if False:
                for i in range(10):
                    print('nop')
            expected = 'INTERESTING_FILE'
            if str(event.eventType) != expected:
                raise Exception(f'{event.eventType} != {expected}')
            expected = 'https://spiderfoot.net/example.zip'
            if str(event.data) != expected:
                raise Exception(f'{event.data} != {expected}')
            raise Exception('OK')
        module.notifyListeners = new_notifyListeners.__get__(module, sfp_intfiles)
        event_type = 'ROOT'
        event_data = 'example data'
        event_module = ''
        source_event = ''
        evt = SpiderFootEvent(event_type, event_data, event_module, source_event)
        event_type = 'LINKED_URL_INTERNAL'
        event_data = 'https://spiderfoot.net/example.zip'
        event_module = 'example module'
        source_event = evt
        evt = SpiderFootEvent(event_type, event_data, event_module, source_event)
        with self.assertRaises(Exception) as cm:
            module.handleEvent(evt)
        self.assertEqual('OK', str(cm.exception))

    def test_handleEvent_event_data_internal_url_without_interesting_file_extension_should_not_return_event(self):
        if False:
            while True:
                i = 10
        sf = SpiderFoot(self.default_options)
        module = sfp_intfiles()
        module.setup(sf, dict())
        target_value = 'spiderfoot.net'
        target_type = 'INTERNET_NAME'
        target = SpiderFootTarget(target_value, target_type)
        module.setTarget(target)

        def new_notifyListeners(self, event):
            if False:
                return 10
            raise Exception(f'Raised event {event.eventType}: {event.data}')
        module.notifyListeners = new_notifyListeners.__get__(module, sfp_intfiles)
        event_type = 'ROOT'
        event_data = 'example data'
        event_module = ''
        source_event = ''
        evt = SpiderFootEvent(event_type, event_data, event_module, source_event)
        event_type = 'LINKED_URL_INTERNAL'
        event_data = 'https://spiderfoot.net/example'
        event_module = 'example module'
        source_event = evt
        evt = SpiderFootEvent(event_type, event_data, event_module, source_event)
        result = module.handleEvent(evt)
        self.assertIsNone(result)