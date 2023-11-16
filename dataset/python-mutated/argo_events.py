import json
import os
import sys
import time
import urllib
import uuid
from datetime import datetime
from metaflow.exception import MetaflowException
from metaflow.metaflow_config import ARGO_EVENTS_WEBHOOK_AUTH, ARGO_EVENTS_WEBHOOK_URL, SERVICE_HEADERS

class ArgoEventException(MetaflowException):
    headline = 'Argo Event Exception'

class ArgoEvent(object):
    """
    ArgoEvent is a small event, a message, that can be published to Argo Workflows. The
    event will eventually start all flows which have been previously deployed with `@trigger`
    to wait for this particular named event.

    Parameters
    ----------
    name : str,
        Name of the event
    url : str, optional
        Override the event endpoint from `ARGO_EVENTS_WEBHOOK_URL`.
    payload : Dict, optional
        A set of key-value pairs delivered in this event. Used to set parameters of triggered flows.
    """

    def __init__(self, name, url=ARGO_EVENTS_WEBHOOK_URL, payload=None, access_token=None):
        if False:
            return 10
        self._name = name
        self._url = url
        self._payload = payload or {}
        self._access_token = access_token

    def add_to_payload(self, key, value):
        if False:
            print('Hello World!')
        '\n        Add a key-value pair in the payload. This is typically used to set parameters\n        of triggered flows. Often, `key` is the parameter name you want to set to\n        `value`. Overrides any existing value of `key`.\n\n        Parameters\n        ----------\n        key : str\n            Key\n        value : str\n            Value\n        '
        self._payload[key] = str(value)
        return self

    def safe_publish(self, payload=None, ignore_errors=True):
        if False:
            i = 10
            return i + 15
        '\n        Publishes an event when called inside a deployed workflow. Outside a deployed workflow\n        this function does nothing.\n\n        Use this function inside flows to create events safely. As this function is a no-op\n        for local runs, you can safely call it during local development without causing unintended\n        side-effects. It takes effect only when deployed on Argo Workflows.\n\n        Parameters\n        ----------\n        payload : dict\n            Additional key-value pairs to add to the payload.\n        ignore_errors : bool, default: True\n            If True, events are created on a best effort basis - errors are silently ignored.\n        '
        return self.publish(payload=payload, force=False, ignore_errors=ignore_errors)

    def publish(self, payload=None, force=True, ignore_errors=True):
        if False:
            while True:
                i = 10
        '\n        Publishes an event.\n\n        Note that the function returns immediately after the event has been sent. It\n        does not wait for flows to start, nor it guarantees that any flows will start.\n\n        Parameters\n        ----------\n        payload : dict\n            Additional key-value pairs to add to the payload.\n        ignore_errors : bool, default: True\n            If True, events are created on a best effort basis - errors are silently ignored.\n        '
        if payload == None:
            payload = {}
        if force or os.environ.get('ARGO_WORKFLOW_TEMPLATE'):
            try:
                headers = {}
                if self._access_token:
                    headers = {'Authorization': 'Bearer {}'.format(self._access_token)}
                if ARGO_EVENTS_WEBHOOK_AUTH == 'service':
                    headers.update(SERVICE_HEADERS)
                data = {'name': self._name, 'payload': {'name': self._name, 'id': str(uuid.uuid4()), 'timestamp': int(time.time()), 'utc_date': datetime.utcnow().strftime('%Y%m%d'), 'generated-by-metaflow': True, **self._payload, **payload}}
                request = urllib.request.Request(self._url, method='POST', headers={'Content-Type': 'application/json', **headers}, data=json.dumps(data).encode('utf-8'))
                retries = 3
                backoff_factor = 2
                for i in range(retries):
                    try:
                        urllib.request.urlopen(request, timeout=10.0)
                        print('Argo Event (%s) published.' % self._name, file=sys.stderr)
                        return data['payload']['id']
                    except urllib.error.HTTPError as e:
                        raise e
                    except urllib.error.URLError as e:
                        if i == retries - 1:
                            raise e
                        else:
                            time.sleep(backoff_factor ** i)
            except Exception as e:
                msg = 'Unable to publish Argo Event (%s): %s' % (self._name, e)
                if ignore_errors:
                    print(msg, file=sys.stderr)
                else:
                    raise ArgoEventException(msg)
        else:
            msg = ('Argo Event (%s) was not published. Use ' + 'ArgoEvent(...).publish(...) ' + 'to force publish.') % self._name
            if ignore_errors:
                print(msg, file=sys.stderr)
            else:
                raise ArgoEventException(msg)