import pytest
import unittest
from modules.sfp_company import sfp_company
from sflib import SpiderFoot
from spiderfoot import SpiderFootEvent, SpiderFootTarget

@pytest.mark.usefixtures
class TestModuleCompany(unittest.TestCase):

    def test_opts(self):
        if False:
            print('Hello World!')
        module = sfp_company()
        self.assertEqual(len(module.opts), len(module.optdescs))

    def test_setup(self):
        if False:
            while True:
                i = 10
        sf = SpiderFoot(self.default_options)
        module = sfp_company()
        module.setup(sf, dict())

    def test_watchedEvents_should_return_list(self):
        if False:
            while True:
                i = 10
        module = sfp_company()
        self.assertIsInstance(module.watchedEvents(), list)

    def test_producedEvents_should_return_list(self):
        if False:
            while True:
                i = 10
        module = sfp_company()
        self.assertIsInstance(module.producedEvents(), list)

    @unittest.skip('todo')
    def test_handleEvent_event_data_ssl_certificate_issued_containing_company_name_should_return_event(self):
        if False:
            return 10
        sf = SpiderFoot(self.default_options)
        module = sfp_company()
        module.setup(sf, dict())
        target_value = 'spiderfoot.net'
        target_type = 'INTERNET_NAME'
        target = SpiderFootTarget(target_value, target_type)
        module.setTarget(target)

        def new_notifyListeners(self, event):
            if False:
                while True:
                    i = 10
            expected = 'COMPANY_NAME'
            if str(event.eventType) != expected:
                raise Exception(f'{event.eventType} != {expected}')
            expected = 'SpiderFoot Corporation'
            if str(event.data) != expected:
                raise Exception(f'{event.data} != {expected}')
            raise Exception('OK')
        module.notifyListeners = new_notifyListeners.__get__(module, sfp_company)
        event_type = 'ROOT'
        event_data = 'example data'
        event_module = ''
        source_event = ''
        evt = SpiderFootEvent(event_type, event_data, event_module, source_event)
        event_type = 'SSL_CERTIFICATE_ISSUED'
        event_data = 'O=SpiderFoot Corporation'
        event_module = 'example module'
        source_event = evt
        evt = SpiderFootEvent(event_type, event_data, event_module, source_event)
        with self.assertRaises(Exception) as cm:
            module.handleEvent(evt)
        self.assertEqual('OK', str(cm.exception))

    @unittest.skip('todo')
    def test_handleEvent_event_data_domain_whois_containing_company_name_should_return_event(self):
        if False:
            i = 10
            return i + 15
        sf = SpiderFoot(self.default_options)
        module = sfp_company()
        module.setup(sf, dict())
        target_value = 'spiderfoot.net'
        target_type = 'INTERNET_NAME'
        target = SpiderFootTarget(target_value, target_type)
        module.setTarget(target)

        def new_notifyListeners(self, event):
            if False:
                while True:
                    i = 10
            expected = 'COMPANY_NAME'
            if str(event.eventType) != expected:
                raise Exception(f'{event.eventType} != {expected}')
            expected = 'SpiderFoot Corporation'
            if str(event.data) != expected:
                raise Exception(f'{event.data} != {expected}')
            raise Exception('OK')
        module.notifyListeners = new_notifyListeners.__get__(module, sfp_company)
        event_type = 'ROOT'
        event_data = 'example data'
        event_module = ''
        source_event = ''
        evt = SpiderFootEvent(event_type, event_data, event_module, source_event)
        event_type = 'DOMAIN_WHOIS'
        event_data = 'Registrant Organization: SpiderFoot Corporation'
        event_module = 'example module'
        source_event = evt
        evt = SpiderFootEvent(event_type, event_data, event_module, source_event)
        with self.assertRaises(Exception) as cm:
            module.handleEvent(evt)
        self.assertEqual('OK', str(cm.exception))

    @unittest.skip('todo')
    def test_handleEvent_event_data_target_web_content_containing_company_name_should_return_event(self):
        if False:
            return 10
        sf = SpiderFoot(self.default_options)
        module = sfp_company()
        module.setup(sf, dict())
        target_value = 'spiderfoot.net'
        target_type = 'INTERNET_NAME'
        target = SpiderFootTarget(target_value, target_type)
        module.setTarget(target)

        def new_notifyListeners(self, event):
            if False:
                print('Hello World!')
            expected = 'COMPANY_NAME'
            if str(event.eventType) != expected:
                raise Exception(f'{event.eventType} != {expected}')
            expected = 'SpiderFoot Corporation'
            if str(event.data) != expected:
                raise Exception(f'{event.data} != {expected}')
            raise Exception('OK')
        module.notifyListeners = new_notifyListeners.__get__(module, sfp_company)
        event_type = 'ROOT'
        event_data = 'example data'
        event_module = ''
        source_event = ''
        evt = SpiderFootEvent(event_type, event_data, event_module, source_event)
        event_type = 'TARGET_WEB_CONTENT'
        event_data = '<p>Copyright SpiderFoot Corporation 2022.</p>'
        event_module = 'example module'
        source_event = evt
        evt = SpiderFootEvent(event_type, event_data, event_module, source_event)
        with self.assertRaises(Exception) as cm:
            module.handleEvent(evt)
        self.assertEqual('OK', str(cm.exception))