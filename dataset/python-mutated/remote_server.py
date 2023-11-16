import logging
import urllib
from typing import Any, Dict, List, Mapping, Tuple, Union
import orjson
import requests
from django.conf import settings
from django.forms.models import model_to_dict
from django.utils.translation import gettext as _
from analytics.models import InstallationCount, RealmCount
from version import ZULIP_VERSION
from zerver.lib.exceptions import JsonableError
from zerver.lib.export import floatify_datetime_fields
from zerver.lib.outgoing_http import OutgoingSession
from zerver.models import Realm, RealmAuditLog

class PushBouncerSession(OutgoingSession):

    def __init__(self) -> None:
        if False:
            return 10
        super().__init__(role='push_bouncer', timeout=30)

class PushNotificationBouncerError(Exception):
    pass

class PushNotificationBouncerRetryLaterError(JsonableError):
    http_status_code = 502

def send_to_push_bouncer(method: str, endpoint: str, post_data: Union[bytes, Mapping[str, Union[str, int, None, bytes]]], extra_headers: Mapping[str, str]={}) -> Dict[str, object]:
    if False:
        return 10
    "While it does actually send the notice, this function has a lot of\n    code and comments around error handling for the push notifications\n    bouncer.  There are several classes of failures, each with its own\n    potential solution:\n\n    * Network errors with requests.request.  We raise an exception to signal\n      it to the callers.\n\n    * 500 errors from the push bouncer or other unexpected responses;\n      we don't try to parse the response, but do make clear the cause.\n\n    * 400 errors from the push bouncer.  Here there are 2 categories:\n      Our server failed to connect to the push bouncer (should throw)\n      vs. client-side errors like an invalid token.\n\n    "
    assert settings.PUSH_NOTIFICATION_BOUNCER_URL is not None
    assert settings.ZULIP_ORG_ID is not None
    assert settings.ZULIP_ORG_KEY is not None
    url = urllib.parse.urljoin(settings.PUSH_NOTIFICATION_BOUNCER_URL, '/api/v1/remotes/' + endpoint)
    api_auth = requests.auth.HTTPBasicAuth(settings.ZULIP_ORG_ID, settings.ZULIP_ORG_KEY)
    headers = {'User-agent': f'ZulipServer/{ZULIP_VERSION}'}
    headers.update(extra_headers)
    try:
        res = PushBouncerSession().request(method, url, data=post_data, auth=api_auth, verify=True, headers=headers)
    except (requests.exceptions.Timeout, requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
        raise PushNotificationBouncerRetryLaterError(f'{type(e).__name__} while trying to connect to push notification bouncer')
    if res.status_code >= 500:
        error_msg = 'Received 500 from push notification bouncer'
        logging.warning(error_msg)
        raise PushNotificationBouncerRetryLaterError(error_msg)
    elif res.status_code >= 400:
        result_dict = orjson.loads(res.content)
        msg = result_dict['msg']
        if 'code' in result_dict and result_dict['code'] == 'INVALID_ZULIP_SERVER':
            raise PushNotificationBouncerError(_('Push notifications bouncer error: {error}').format(error=msg))
        elif endpoint == 'push/test_notification' and 'code' in result_dict and (result_dict['code'] == 'INVALID_REMOTE_PUSH_DEVICE_TOKEN'):
            from zerver.lib.push_notifications import InvalidRemotePushDeviceTokenError
            raise InvalidRemotePushDeviceTokenError
        else:
            raise JsonableError(msg)
    elif res.status_code != 200:
        raise PushNotificationBouncerError(f'Push notification bouncer returned unexpected status code {res.status_code}')
    return orjson.loads(res.content)

def send_json_to_push_bouncer(method: str, endpoint: str, post_data: Mapping[str, object]) -> Dict[str, object]:
    if False:
        print('Hello World!')
    return send_to_push_bouncer(method, endpoint, orjson.dumps(post_data), extra_headers={'Content-type': 'application/json'})
REALMAUDITLOG_PUSHED_FIELDS = ['id', 'realm', 'event_time', 'backfilled', 'extra_data', 'event_type']

def build_analytics_data(realm_count_query: Any, installation_count_query: Any, realmauditlog_query: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if False:
        while True:
            i = 10
    MAX_CLIENT_BATCH_SIZE = 10000
    data = {}
    data['analytics_realmcount'] = [model_to_dict(row) for row in realm_count_query.order_by('id')[0:MAX_CLIENT_BATCH_SIZE]]
    data['analytics_installationcount'] = [model_to_dict(row) for row in installation_count_query.order_by('id')[0:MAX_CLIENT_BATCH_SIZE]]
    data['zerver_realmauditlog'] = [model_to_dict(row, fields=REALMAUDITLOG_PUSHED_FIELDS) for row in realmauditlog_query.order_by('id')[0:MAX_CLIENT_BATCH_SIZE]]
    floatify_datetime_fields(data, 'analytics_realmcount')
    floatify_datetime_fields(data, 'analytics_installationcount')
    floatify_datetime_fields(data, 'zerver_realmauditlog')
    return (data['analytics_realmcount'], data['analytics_installationcount'], data['zerver_realmauditlog'])

def get_realms_info_for_push_bouncer() -> List[Dict[str, Any]]:
    if False:
        print('Hello World!')
    realms = Realm.objects.order_by('id')
    realm_info_dicts = [dict(id=realm.id, uuid=str(realm.uuid), uuid_owner_secret=realm.uuid_owner_secret, host=realm.host, url=realm.uri, deactivated=realm.deactivated, date_created=realm.date_created.timestamp()) for realm in realms]
    return realm_info_dicts

def send_analytics_to_push_bouncer() -> None:
    if False:
        i = 10
        return i + 15
    try:
        result = send_to_push_bouncer('GET', 'server/analytics/status', {})
    except PushNotificationBouncerRetryLaterError as e:
        logging.warning(e.msg, exc_info=True)
        return
    last_acked_realm_count_id = result['last_realm_count_id']
    last_acked_installation_count_id = result['last_installation_count_id']
    last_acked_realmauditlog_id = result['last_realmauditlog_id']
    (realm_count_data, installation_count_data, realmauditlog_data) = build_analytics_data(realm_count_query=RealmCount.objects.filter(id__gt=last_acked_realm_count_id), installation_count_query=InstallationCount.objects.filter(id__gt=last_acked_installation_count_id), realmauditlog_query=RealmAuditLog.objects.filter(event_type__in=RealmAuditLog.SYNCED_BILLING_EVENTS, id__gt=last_acked_realmauditlog_id))
    if len(realm_count_data) + len(installation_count_data) + len(realmauditlog_data) == 0:
        return
    request = {'realm_counts': orjson.dumps(realm_count_data).decode(), 'installation_counts': orjson.dumps(installation_count_data).decode(), 'realmauditlog_rows': orjson.dumps(realmauditlog_data).decode(), 'realms': orjson.dumps(get_realms_info_for_push_bouncer()).decode(), 'version': orjson.dumps(ZULIP_VERSION).decode()}
    try:
        send_to_push_bouncer('POST', 'server/analytics', request)
    except JsonableError as e:
        logging.warning(e.msg)