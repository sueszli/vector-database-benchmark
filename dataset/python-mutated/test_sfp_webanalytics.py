import pytest
import unittest
from modules.sfp_webanalytics import sfp_webanalytics
from sflib import SpiderFoot
from spiderfoot import SpiderFootEvent, SpiderFootTarget

@pytest.mark.usefixtures
class TestModuleWebAnalytics(unittest.TestCase):

    def test_opts(self):
        if False:
            print('Hello World!')
        module = sfp_webanalytics()
        self.assertEqual(len(module.opts), len(module.optdescs))

    def test_setup(self):
        if False:
            for i in range(10):
                print('nop')
        sf = SpiderFoot(self.default_options)
        module = sfp_webanalytics()
        module.setup(sf, dict())

    def test_watchedEvents_should_return_list(self):
        if False:
            print('Hello World!')
        module = sfp_webanalytics()
        self.assertIsInstance(module.watchedEvents(), list)

    def test_producedEvents_should_return_list(self):
        if False:
            print('Hello World!')
        module = sfp_webanalytics()
        self.assertIsInstance(module.producedEvents(), list)

    def test_handleEvent_event_data_target_web_content_containing_web_analytics_string_should_return_event(self):
        if False:
            i = 10
            return i + 15
        sf = SpiderFoot(self.default_options)
        module = sfp_webanalytics()
        module.setup(sf, dict())
        target_value = 'spiderfoot.net'
        target_type = 'INTERNET_NAME'
        target = SpiderFootTarget(target_value, target_type)
        module.setTarget(target)

        def new_notifyListeners(self, event):
            if False:
                while True:
                    i = 10
            expected = 'WEB_ANALYTICS_ID'
            if str(event.eventType) != expected:
                raise Exception(f'{event.eventType} != {expected}')
            expected = 'Google Analytics: ua-1111111111-123'
            if str(event.data) != expected:
                raise Exception(f'{event.data} != {expected}')
            raise Exception('OK')
        module.notifyListeners = new_notifyListeners.__get__(module, sfp_webanalytics)
        event_type = 'ROOT'
        event_data = 'example data'
        event_module = ''
        source_event = ''
        evt = SpiderFootEvent(event_type, event_data, event_module, source_event)
        event_type = 'TARGET_WEB_CONTENT'
        event_data = '<p>example data ua-1111111111-123 example data</p>'
        event_module = 'example module'
        source_event = evt
        evt = SpiderFootEvent(event_type, event_data, event_module, source_event)
        with self.assertRaises(Exception) as cm:
            module.handleEvent(evt)
        self.assertEqual('OK', str(cm.exception))

    def test_handleEvent_event_data_target_web_content_not_containing_web_analytics_string_should_not_create_event(self):
        if False:
            return 10
        sf = SpiderFoot(self.default_options)
        module = sfp_webanalytics()
        module.setup(sf, dict())
        target_value = 'spiderfoot.net'
        target_type = 'INTERNET_NAME'
        target = SpiderFootTarget(target_value, target_type)
        module.setTarget(target)

        def new_notifyListeners(self, event):
            if False:
                return 10
            raise Exception(f'Raised event {event.eventType}: {event.data}')
        module.notifyListeners = new_notifyListeners.__get__(module, sfp_webanalytics)
        event_type = 'ROOT'
        event_data = 'example data'
        event_module = ''
        source_event = ''
        evt = SpiderFootEvent(event_type, event_data, event_module, source_event)
        event_type = 'TARGET_WEB_CONTENT'
        event_data = 'example data'
        event_module = 'example module'
        source_event = evt
        evt = SpiderFootEvent(event_type, event_data, event_module, source_event)
        result = module.handleEvent(evt)
        self.assertIsNone(result)

    def test_handleEvent_event_dns_text_containing_web_analytics_string_should_return_event(self):
        if False:
            while True:
                i = 10
        sf = SpiderFoot(self.default_options)
        module = sfp_webanalytics()
        module.setup(sf, dict())
        target_value = 'spiderfoot.net'
        target_type = 'INTERNET_NAME'
        target = SpiderFootTarget(target_value, target_type)
        module.setTarget(target)

        def new_notifyListeners(self, event):
            if False:
                while True:
                    i = 10
            expected = 'WEB_ANALYTICS_ID'
            if str(event.eventType) != expected:
                raise Exception(f'{event.eventType} != {expected}')
            expected = 'Google Site Verification: abcdefghijklmnopqrstuvwxyz1234567890abc_def'
            if str(event.data) != expected:
                raise Exception(f'{event.data} != {expected}')
            raise Exception('OK')
        module.notifyListeners = new_notifyListeners.__get__(module, sfp_webanalytics)
        event_type = 'ROOT'
        event_data = 'example data'
        event_module = ''
        source_event = ''
        evt = SpiderFootEvent(event_type, event_data, event_module, source_event)
        event_type = 'DNS_TEXT'
        event_data = 'google-site-verification=abcdefghijklmnopqrstuvwxyz1234567890abc_def'
        event_module = 'example module'
        source_event = evt
        evt = SpiderFootEvent(event_type, event_data, event_module, source_event)
        with self.assertRaises(Exception) as cm:
            module.handleEvent(evt)
        self.assertEqual('OK', str(cm.exception))

    def test_handleEvent_event_data_dns_text_not_containing_web_analytics_string_should_not_create_event(self):
        if False:
            return 10
        sf = SpiderFoot(self.default_options)
        module = sfp_webanalytics()
        module.setup(sf, dict())
        target_value = 'spiderfoot.net'
        target_type = 'INTERNET_NAME'
        target = SpiderFootTarget(target_value, target_type)
        module.setTarget(target)

        def new_notifyListeners(self, event):
            if False:
                return 10
            raise Exception(f'Raised event {event.eventType}: {event.data}')
        module.notifyListeners = new_notifyListeners.__get__(module, sfp_webanalytics)
        event_type = 'ROOT'
        event_data = 'example data'
        event_module = ''
        source_event = ''
        evt = SpiderFootEvent(event_type, event_data, event_module, source_event)
        event_type = 'DNS_TEXT'
        event_data = 'example data'
        event_module = 'example module'
        source_event = evt
        evt = SpiderFootEvent(event_type, event_data, event_module, source_event)
        result = module.handleEvent(evt)
        self.assertIsNone(result)