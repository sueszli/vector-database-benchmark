import json
import logging
import urllib.request
import time
import datetime
import random
from ga4mp.utils import params_dict
from ga4mp.event import Event
from ga4mp.store import BaseStore, DictStore
import os, sys
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), '..')))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class BaseGa4mp(object):
    """
    Parent class that provides an interface for sending data to Google Analytics, supporting the GA4 Measurement Protocol.

    Parameters
    ----------
    api_secret : string
        Generated through the Google Analytics UI. To create a new secret, navigate in the Google Analytics UI to: Admin > Data Streams >
        [choose your stream] > Measurement Protocol API Secrets > Create

    See Also
    --------

    * Measurement Protocol (Google Analytics 4): https://developers.google.com/analytics/devguides/collection/protocol/ga4

    Examples
    --------
    # Initialize tracking object for gtag usage
    >>> ga = gtagMP(api_secret = "API_SECRET", measurement_id = "MEASUREMENT_ID", client_id="CLIENT_ID")

    # Initialize tracking object for Firebase usage
    >>> ga = firebaseMP(api_secret = "API_SECRET", firebase_app_id = "FIREBASE_APP_ID", app_instance_id="APP_INSTANCE_ID")

    # Build an event
    >>> event_type = 'new_custom_event'
    >>> event_parameters = {'parameter_key_1': 'parameter_1', 'parameter_key_2': 'parameter_2'}
    >>> event = {'name': event_type, 'params': event_parameters }
    >>> events = [event]

    # Send a custom event to GA4 immediately
    >>> ga.send(events)

    # Postponed send of a custom event to GA4
    >>> ga.send(events, postpone=True)
    >>> ga.postponed_send()
    """

    def __init__(self, api_secret, store: BaseStore=None):
        if False:
            for i in range(10):
                print('nop')
        self._initialization_time = time.time()
        self.api_secret = api_secret
        self._event_list = []
        assert store is None or isinstance(store, BaseStore), 'if supplied, store must be an instance of BaseStore'
        self.store = store or DictStore()
        self._check_store_requirements()
        self._base_domain = 'https://www.google-analytics.com/mp/collect'
        self._validation_domain = 'https://www.google-analytics.com/debug/mp/collect'

    def _check_store_requirements(self):
        if False:
            print('Hello World!')
        if self.store.get_session_parameter('session_id') is None:
            self.store.set_session_parameter(name='session_id', value=int(self._initialization_time))
        self.store.set_session_parameter(name='last_interaction_time_msec', value=int(self._initialization_time * 1000))

    def create_new_event(self, name):
        if False:
            while True:
                i = 10
        return Event(name=name)

    def send(self, events, validation_hit=False, postpone=False, date=None):
        if False:
            while True:
                i = 10
        "\n        Method to send an http post request to google analytics with the specified events.\n\n        Parameters\n        ----------\n        events : List[Dict]\n            A list of dictionaries of the events to be sent to Google Analytics. The list of dictionaries should adhere\n            to the following format:\n\n            [{'name': 'level_end',\n            'params' : {'level_name': 'First',\n                        'success': 'True'}\n            },\n            {'name': 'level_up',\n            'params': {'character': 'John Madden',\n                        'level': 'First'}\n            }]\n\n        validation_hit : bool, optional\n            Boolean to depict if events should be tested against the Measurement Protocol Validation Server, by default False\n        postpone : bool, optional\n            Boolean to depict if provided event list should be postponed, by default False\n        date : datetime\n            Python datetime object for sending a historical event at the given date. Date cannot be in the future.\n        "
        self._check_params(events)
        self._check_date_not_in_future(date)
        self._add_session_id_and_engagement_time(events)
        if postpone is True:
            for event in events:
                event['_timestamp_micros'] = self._get_timestamp(time.time())
                self._event_list.append(event)
        else:
            batched_event_list = [events[event:event + 25] for event in range(0, len(events), 25)]
            self._http_post(batched_event_list, validation_hit=validation_hit, date=date)

    def postponed_send(self):
        if False:
            print('Hello World!')
        '\n        Method to send the events provided to Ga4mp.send(events,postpone=True)\n        '
        for event in self._event_list:
            self._http_post([event], postpone=True)
        self._event_list = []

    def append_event_to_params_dict(self, new_name_and_parameters):
        if False:
            print('Hello World!')
        "\n        Method to append event name and parameters key-value pairing(s) to parameters dictionary.\n\n        Parameters\n        ----------\n        new_name_and_parameters : Dict\n            A dictionary with one key-value pair representing a new type of event to be sent to Google Analytics.\n            The dictionary should adhere to the following format:\n\n            {'new_name': ['new_param_1', 'new_param_2', 'new_param_3']}\n        "
        params_dict.update(new_name_and_parameters)

    def _http_post(self, batched_event_list, validation_hit=False, postpone=False, date=None):
        if False:
            print('Hello World!')
        '\n        Method to send http POST request to google-analytics.\n\n        Parameters\n        ----------\n        batched_event_list : List[List[Dict]]\n            List of List of events. Places initial event payload into a list to send http POST in batches.\n        validation_hit : bool, optional\n            Boolean to depict if events should be tested against the Measurement Protocol Validation Server, by default False\n        postpone : bool, optional\n            Boolean to depict if provided event list should be postponed, by default False\n        date : datetime\n            Python datetime object for sending a historical event at the given date. Date cannot be in the future.\n            Timestamp micros supports up to 48 hours of backdating.\n            If date is specified, postpone must be False or an assertion will be thrown.\n        '
        self._check_date_not_in_future(date)
        status_code = None
        domain = self._base_domain
        if validation_hit is True:
            domain = self._validation_domain
        logger.info(f'Sending POST to: {domain}')
        batch_number = 1
        for batch in batched_event_list:
            url = self._build_url(domain=domain)
            request = self._build_request(batch=batch)
            self._add_user_props_to_hit(request)
            request['events'] = {'name': batch['name'], 'params': batch['params']} if postpone else batch
            if date is not None:
                logger.info(f'Setting event timestamp to: {date}')
                assert postpone is False, 'Cannot send postponed historical hit, ensure postpone=False'
                ts = self._datetime_to_timestamp(date)
                ts_micro = self._get_timestamp(ts)
                request['timestamp_micros'] = int(ts_micro)
                logger.info(f"Timestamp of request is: {request['timestamp_micros']}")
            if postpone:
                request['timestamp_micros'] = batch['_timestamp_micros']
            req = urllib.request.Request(url)
            req.add_header('Content-Type', 'application/json; charset=utf-8')
            jsondata = json.dumps(request)
            json_data_as_bytes = jsondata.encode('utf-8')
            req.add_header('Content-Length', len(json_data_as_bytes))
            result = urllib.request.urlopen(req, json_data_as_bytes)
            status_code = result.status
            logger.info(f'Batch Number: {batch_number}')
            logger.info(f'Status code: {status_code}')
            batch_number += 1
        return status_code

    def _check_params(self, events):
        if False:
            i = 10
            return i + 15
        "\n        Method to check whether the provided event payload parameters align with supported parameters.\n\n        Parameters\n        ----------\n        events : List[Dict]\n            A list of dictionaries of the events to be sent to Google Analytics. The list of dictionaries should adhere\n            to the following format:\n\n            [{'name': 'level_end',\n            'params' : {'level_name': 'First',\n                        'success': 'True'}\n            },\n            {'name': 'level_up',\n            'params': {'character': 'John Madden',\n                        'level': 'First'}\n            }]\n        "
        assert type(events) == list, 'events should be a list'
        for event in events:
            assert isinstance(event, dict), 'each event should be an instance of a dictionary'
            assert 'name' in event, 'each event should have a "name" key'
            assert 'params' in event, 'each event should have a "params" key'
        for e in events:
            event_name = e['name']
            event_params = e['params']
            if event_name in params_dict.keys():
                for parameter in params_dict[event_name]:
                    if parameter not in event_params.keys():
                        logger.warning(f"WARNING: Event parameters do not match event type.\nFor {event_name} event type, the correct parameter(s) are {params_dict[event_name]}.\nThe parameter '{parameter}' triggered this warning.\nFor a breakdown of currently supported event types and their parameters go here: https://support.google.com/analytics/answer/9267735\n")

    def _add_session_id_and_engagement_time(self, events):
        if False:
            print('Hello World!')
        '\n        Method to add the session_id and engagement_time_msec parameter to all events.\n        '
        for event in events:
            current_time_in_milliseconds = int(time.time() * 1000)
            event_params = event['params']
            if 'session_id' not in event_params.keys():
                event_params['session_id'] = self.store.get_session_parameter('session_id')
            if 'engagement_time_msec' not in event_params.keys():
                last_interaction_time = self.store.get_session_parameter('last_interaction_time_msec')
                event_params['engagement_time_msec'] = current_time_in_milliseconds - last_interaction_time if current_time_in_milliseconds > last_interaction_time else 0
                self.store.set_session_parameter(name='last_interaction_time_msec', value=current_time_in_milliseconds)

    def _add_user_props_to_hit(self, hit):
        if False:
            print('Hello World!')
        '\n        Method is a helper function to add user properties to outgoing hits.\n\n        Parameters\n        ----------\n        hit : dict\n        '
        for key in self.store.get_all_user_properties():
            try:
                if key in ['user_id', 'non_personalized_ads']:
                    hit.update({key: self.store.get_user_property(key)})
                else:
                    if 'user_properties' not in hit.keys():
                        hit.update({'user_properties': {}})
                    hit['user_properties'].update({key: {'value': self.store.get_user_property(key)}})
            except:
                logger.info(f'Failed to add user property to outgoing hit: {key}')

    def _get_timestamp(self, timestamp):
        if False:
            for i in range(10):
                print('nop')
        '\n        Method returns UNIX timestamp in microseconds for postponed hits.\n\n        Parameters\n        ----------\n        None\n        '
        return int(timestamp * 1000000.0)

    def _datetime_to_timestamp(self, dt):
        if False:
            for i in range(10):
                print('nop')
        '\n        Private method to convert a datetime object into a timestamp\n\n        Parameters\n        ----------\n        dt : datetime\n            A datetime object in any format\n\n        Returns\n        -------\n        timestamp\n            A UNIX timestamp in milliseconds\n        '
        return time.mktime(dt.timetuple())

    def _check_date_not_in_future(self, date):
        if False:
            while True:
                i = 10
        '\n        Method to check that provided date is not in the future.\n\n        Parameters\n        ----------\n        date : datetime\n            Python datetime object\n        '
        if date is None:
            pass
        else:
            assert date <= datetime.datetime.now(), 'Provided date cannot be in the future'

    def _build_url(self, domain):
        if False:
            while True:
                i = 10
        raise NotImplementedError('Subclass should be using this function, but it was called through the base class instead.')

    def _build_request(self, batch):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('Subclass should be using this function, but it was called through the base class instead.')

class GtagMP(BaseGa4mp):
    """
    Subclass for users of gtag. See `Ga4mp` parent class for examples.

    Parameters
    ----------
    measurement_id : string
        The identifier for a Data Stream. Found in the Google Analytics UI under: Admin > Data Streams > [choose your stream] > Measurement ID (top-right)
    client_id : string
        A unique identifier for a client, representing a specific browser/device.
    """

    def __init__(self, api_secret, measurement_id, client_id):
        if False:
            i = 10
            return i + 15
        super().__init__(api_secret)
        self.measurement_id = measurement_id
        self.client_id = client_id

    def _build_url(self, domain):
        if False:
            i = 10
            return i + 15
        return f'{domain}?measurement_id={self.measurement_id}&api_secret={self.api_secret}'

    def _build_request(self, batch):
        if False:
            while True:
                i = 10
        return {'client_id': self.client_id, 'events': batch}

    def random_client_id(self):
        if False:
            i = 10
            return i + 15
        '\n        Utility function for generating a new client ID matching the typical format of 10 random digits and the UNIX timestamp in seconds, joined by a period.\n        '
        return '%0.10d' % random.randint(0, 9999999999) + '.' + str(int(time.time()))

class FirebaseMP(BaseGa4mp):
    """
    Subclass for users of Firebase. See `Ga4mp` parent class for examples.

    Parameters
    ----------
    firebase_app_id : string
        The identifier for a Firebase app. Found in the Firebase console under: Project Settings > General > Your Apps > App ID.
    app_instance_id : string
        A unique identifier for a Firebase app instance.
            * Android - getAppInstanceId() - https://firebase.google.com/docs/reference/android/com/google/firebase/analytics/FirebaseAnalytics#public-taskstring-getappinstanceid
            * Kotlin - getAppInstanceId() - https://firebase.google.com/docs/reference/kotlin/com/google/firebase/analytics/FirebaseAnalytics#getappinstanceid
            * Swift - appInstanceID() - https://firebase.google.com/docs/reference/swift/firebaseanalytics/api/reference/Classes/Analytics#appinstanceid
            * Objective-C - appInstanceID - https://firebase.google.com/docs/reference/ios/firebaseanalytics/api/reference/Classes/FIRAnalytics#+appinstanceid
            * C++ - GetAnalyticsInstanceId() - https://firebase.google.com/docs/reference/cpp/namespace/firebase/analytics#getanalyticsinstanceid
            * Unity - GetAnalyticsInstanceIdAsync() - https://firebase.google.com/docs/reference/unity/class/firebase/analytics/firebase-analytics#getanalyticsinstanceidasync
    """

    def __init__(self, api_secret, firebase_app_id, app_instance_id):
        if False:
            i = 10
            return i + 15
        super().__init__(api_secret)
        self.firebase_app_id = firebase_app_id
        self.app_instance_id = app_instance_id

    def _build_url(self, domain):
        if False:
            i = 10
            return i + 15
        return f'{domain}?firebase_app_id={self.firebase_app_id}&api_secret={self.api_secret}'

    def _build_request(self, batch):
        if False:
            print('Hello World!')
        return {'app_instance_id': self.app_instance_id, 'events': batch}