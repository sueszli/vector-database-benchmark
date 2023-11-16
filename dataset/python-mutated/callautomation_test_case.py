import json
import threading
import time
import os
import inspect
from datetime import datetime, timedelta
from typing import Dict, Any, List
import requests
from azure.servicebus import ServiceBusClient
from devtools_testutils import AzureRecordedTestCase, is_live
from devtools_testutils.helpers import get_test_id
from azure.communication.callautomation import CommunicationIdentifierKind, CallAutomationClient, CallConnectionClient, CommunicationIdentifier
from azure.communication.callautomation._shared.models import identifier_from_raw_id
from azure.communication.identity import CommunicationIdentityClient
from azure.communication.phonenumbers import PhoneNumbersClient

class CallAutomationRecordedTestCase(AzureRecordedTestCase):

    @classmethod
    def setup_class(cls):
        if False:
            for i in range(10):
                print('nop')
        if is_live():
            print('Live Test')
            cls.connection_str = os.environ.get('COMMUNICATION_LIVETEST_STATIC_CONNECTION_STRING')
            cls.servicebus_connection_str = os.environ.get('SERVICEBUS_STRING')
            cls.dispatcher_endpoint = os.environ.get('DISPATCHER_ENDPOINT')
            cls.file_source_url = os.environ.get('FILE_SOURCE_URL')
        else:
            print('Recorded Test')
            cls.connection_str = 'endpoint=https://someEndpoint/;accesskey=someAccessKeyw=='
            cls.servicebus_connection_str = 'Endpoint=sb://someEndpoint/;SharedAccessKeyName=somekey;SharedAccessKey=someAccessKey='
            cls.dispatcher_endpoint = 'https://REDACTED.azurewebsites.net'
            cls.file_source_url = 'https://REDACTED/prompt.wav'
        cls.dispatcher_callback = cls.dispatcher_endpoint + '/api/servicebuscallback/events'
        cls.identity_client = CommunicationIdentityClient.from_connection_string(cls.connection_str)
        cls.phonenumber_client = PhoneNumbersClient.from_connection_string(cls.connection_str)
        cls.service_bus_client = ServiceBusClient.from_connection_string(cls.servicebus_connection_str)
        cls.wait_for_event_flags = []
        cls.event_store: Dict[str, Dict[str, Any]] = {}
        cls.event_to_save: Dict[str, Dict[str, Any]] = {}
        cls.open_call_connections: Dict[str, CallConnectionClient] = {}

    @classmethod
    def teardown_class(cls):
        if False:
            i = 10
            return i + 15
        cls.wait_for_event_flags.clear()
        for cc in cls.open_call_connections.values():
            cc.hang_up(is_for_everyone=True)

    def setup_method(self, method):
        if False:
            for i in range(10):
                print('nop')
        self.test_name = get_test_id().split('/')[-1]
        self.event_store: Dict[str, Dict[str, Any]] = {}
        self.event_to_save: Dict[str, Dict[str, Any]] = {}
        self._prepare_events_recording()
        pass

    def teardown_method(self, method):
        if False:
            return 10
        self._record_method_events()
        self.event_store: Dict[str, Dict[str, Any]] = {}
        self.event_to_save: Dict[str, Dict[str, Any]] = {}
        pass

    @staticmethod
    def _format_string(s) -> str:
        if False:
            for i in range(10):
                print('nop')
        s1 = f'{s[:12]}-{s[12:16]}-{s[16:20]}-{s[20:24]}-{s[24:36]}'
        s2 = f'{s[36:44]}-{s[44:48]}-{s[48:52]}-{s[52:56]}-{s[56:]}'
        return f'{s1}_{s2}'

    @staticmethod
    def _format_phonenumber_string(s) -> str:
        if False:
            return 10
        return s.replace(':+', 'u002B')

    @staticmethod
    def _parse_ids_from_identifier(identifier: CommunicationIdentifier) -> str:
        if False:
            i = 10
            return i + 15
        if identifier is None:
            raise ValueError('Identifier cannot be None')
        elif identifier.kind == CommunicationIdentifierKind.COMMUNICATION_USER:
            return CallAutomationRecordedTestCase._format_string(''.join(filter(str.isalnum, identifier.raw_id)))
        elif identifier.kind == CommunicationIdentifierKind.PHONE_NUMBER:
            return CallAutomationRecordedTestCase._format_phonenumber_string(identifier.raw_id)
        else:
            raise ValueError('Identifier type not supported')

    @staticmethod
    def _event_key_gen(event_type: str) -> str:
        if False:
            while True:
                i = 10
        return event_type

    @staticmethod
    def _unique_key_gen(caller_identifier: CommunicationIdentifier, receiver_identifier: CommunicationIdentifier) -> str:
        if False:
            print('Hello World!')
        return CallAutomationRecordedTestCase._parse_ids_from_identifier(caller_identifier) + CallAutomationRecordedTestCase._parse_ids_from_identifier(receiver_identifier)

    def _get_test_event_file_name(self):
        if False:
            for i in range(10):
                print('nop')
        script_dir = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(script_dir, 'events', f'{self.test_name}.event.json')
        return file_path

    def _message_awaiter(self, unique_id) -> None:
        if False:
            return 10
        service_bus_receiver = self.service_bus_client.get_queue_receiver(queue_name=unique_id)
        while unique_id in self.wait_for_event_flags:
            received_messages = service_bus_receiver.receive_messages(max_wait_time=20)
            for msg in received_messages:
                body_bytes = b''.join(msg.body)
                body_str = body_bytes.decode('utf-8')
                mapper = json.loads(body_str)
                if 'incomingCallContext' in mapper:
                    caller = identifier_from_raw_id(mapper['from']['rawId'])
                    receiver = identifier_from_raw_id(mapper['to']['rawId'])
                    unique_id = self._unique_key_gen(caller, receiver)
                    key = self._event_key_gen('IncomingCall')
                    print('EventRegistration(IncomingCall):' + key)
                    self.event_store[key] = mapper
                    self.event_to_save[key] = mapper
                else:
                    if isinstance(mapper, list):
                        mapper = mapper[0]
                    if mapper['type']:
                        key = self._event_key_gen(mapper['type'].split('.')[-1])
                        print('EventRegistration:' + key)
                        self.event_store[key] = mapper
                        self.event_to_save[key] = mapper
                service_bus_receiver.complete_message(msg)
            time.sleep(1)
        return

    def _prepare_events_recording(self) -> None:
        if False:
            print('Hello World!')
        if not is_live():
            file_path = self._get_test_event_file_name()
            try:
                with open(file_path, 'r') as json_file:
                    self.event_store = json.load(json_file)
            except IOError as e:
                raise SystemExit(f'File write operation failed: {e}')

    def _record_method_events(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if is_live():
            file_path = self._get_test_event_file_name()
            try:
                with open(file_path, 'w') as json_file:
                    json.dump(self.event_to_save, json_file)
            except IOError as e:
                raise SystemExit(f'File write operation failed: {e}')

    def check_for_event(self, event_type: str, call_connection_id: str, wait_time: timedelta) -> Any:
        if False:
            i = 10
            return i + 15
        key = self._event_key_gen(event_type)
        time_out_time = datetime.now() + wait_time
        while datetime.now() < time_out_time:
            popped_event = self.event_store.pop(key, None)
            if popped_event is not None:
                print(f'Matching Event Found [{key}]')
                return popped_event
            time.sleep(1)
        return None

    def establish_callconnection_voip(self, caller, target) -> tuple:
        if False:
            return 10
        call_automation_client_caller = CallAutomationClient.from_connection_string(self.connection_str, source=caller)
        call_automation_client_target = CallAutomationClient.from_connection_string(self.connection_str, source=target)
        unique_id = self._unique_key_gen(caller, target)
        if is_live():
            dispatcher_url = f'{self.dispatcher_endpoint}/api/servicebuscallback/subscribe?q={unique_id}'
            response = requests.post(dispatcher_url)
            if response is None:
                raise ValueError('Response cannot be None')
            print(f'Subscription to dispatcher of {unique_id}: {response.status_code}')
            self.wait_for_event_flags.append(unique_id)
            thread = threading.Thread(target=self._message_awaiter, args=(unique_id,))
            thread.start()
        create_call_result = call_automation_client_caller.create_call(target_participant=target, callback_url=self.dispatcher_callback + '?q={}'.format(unique_id))
        if create_call_result is None:
            raise ValueError('Invalid create_call_result')
        caller_connection_id = create_call_result.call_connection_id
        if caller_connection_id is None:
            raise ValueError('Caller connection ID is None')
        incoming_call_event = self.check_for_event('IncomingCall', unique_id, timedelta(seconds=30))
        if incoming_call_event is None:
            raise ValueError('incoming_call_event is None')
        incoming_call_context = incoming_call_event['incomingCallContext']
        answer_call_result = call_automation_client_target.answer_call(incoming_call_context=incoming_call_context, callback_url=self.dispatcher_callback)
        if answer_call_result is None:
            raise ValueError('Invalid answer_call result')
        call_connection_caller = CallConnectionClient.from_connection_string(self.connection_str, caller_connection_id)
        call_connection_target = CallConnectionClient.from_connection_string(self.connection_str, answer_call_result.call_connection_id)
        self.open_call_connections[unique_id] = call_connection_caller
        return (unique_id, call_connection_caller, call_connection_target)

    def establish_callconnection_pstn(self, caller, target) -> tuple:
        if False:
            i = 10
            return i + 15
        call_automation_client_caller = CallAutomationClient.from_connection_string(self.connection_str)
        call_automation_client_target = CallAutomationClient.from_connection_string(self.connection_str)
        unique_id = self._unique_key_gen(caller, target)
        if is_live():
            dispatcher_url = f'{self.dispatcher_endpoint}/api/servicebuscallback/subscribe?q={unique_id}'
            response = requests.post(dispatcher_url)
            if response is None:
                raise ValueError('Response cannot be None')
            print(f'Subscription to dispatcher of {unique_id}: {response.status_code}')
            self.wait_for_event_flags.append(unique_id)
            thread = threading.Thread(target=self._message_awaiter, args=(unique_id,))
            thread.start()
        create_call_result = call_automation_client_caller.create_call(target_participant=target, source_caller_id_number=caller, callback_url=self.dispatcher_callback + '?q={}'.format(unique_id))
        if create_call_result is None:
            raise ValueError('Invalid create_call_result')
        caller_connection_id = create_call_result.call_connection_id
        if caller_connection_id is None:
            raise ValueError('Caller connection ID is None')
        incoming_call_event = self.check_for_event('IncomingCall', unique_id, timedelta(seconds=30))
        if incoming_call_event is None:
            raise ValueError('incoming_call_event is None')
        incoming_call_context = incoming_call_event['incomingCallContext']
        answer_call_result = call_automation_client_target.answer_call(incoming_call_context=incoming_call_context, callback_url=self.dispatcher_callback)
        if answer_call_result is None:
            raise ValueError('Invalid answer_call result')
        call_connection_caller = CallConnectionClient.from_connection_string(self.connection_str, caller_connection_id)
        call_connection_target = CallConnectionClient.from_connection_string(self.connection_str, answer_call_result.call_connection_id)
        self.open_call_connections[unique_id] = call_connection_caller
        return (unique_id, call_connection_caller, call_connection_target)

    def terminate_call(self, unique_id) -> None:
        if False:
            i = 10
            return i + 15
        try:
            call_connection = self.open_call_connections.pop(unique_id, None)
            if call_connection is not None:
                call_connection.hang_up(is_for_everyone=True)
                disconnected_event = self.check_for_event('CallDisconnected', call_connection._call_connection_id, timedelta(seconds=15))
                if disconnected_event is None:
                    raise ValueError('Receiver CallDisconnected event is None')
        finally:
            while unique_id in self.wait_for_event_flags:
                self.wait_for_event_flags.remove(unique_id)
            pass