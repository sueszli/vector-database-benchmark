import datetime
import random
import string
import time
import google.auth
from google.cloud.retail import ImportUserEventsRequest, UserEvent, UserEventInlineSource, UserEventInputConfig, UserEventServiceClient
from google.protobuf.timestamp_pb2 import Timestamp
project_id = google.auth.default()[1]
default_catalog = f'projects/{project_id}/locations/global/catalogs/default_catalog'

def get_user_events():
    if False:
        while True:
            i = 10
    user_events = []
    for x in range(3):
        timestamp = Timestamp()
        timestamp.seconds = int(datetime.datetime.now().timestamp())
        user_event = UserEvent()
        user_event.event_type = 'home-page-view'
        user_event.visitor_id = ''.join(random.sample(string.ascii_lowercase, 4)) + 'event_' + str(x)
        user_event.event_time = timestamp
        user_events.append(user_event)
    print(user_events)
    return user_events

def get_import_events_inline_source_request(user_events_to_import):
    if False:
        while True:
            i = 10
    inline_source = UserEventInlineSource()
    inline_source.user_events = user_events_to_import
    input_config = UserEventInputConfig()
    input_config.user_event_inline_source = inline_source
    import_request = ImportUserEventsRequest()
    import_request.parent = default_catalog
    import_request.input_config = input_config
    print('---import user events from inline source request---')
    print(import_request)
    return import_request

def import_user_events_from_inline_source():
    if False:
        i = 10
        return i + 15
    import_inline_request = get_import_events_inline_source_request(get_user_events())
    import_operation = UserEventServiceClient().import_user_events(import_inline_request)
    print('---the operation was started:----')
    print(import_operation.operation.name)
    while not import_operation.done():
        print('---please wait till operation is done---')
        time.sleep(5)
    print('---import user events operation is done---')
    if import_operation.metadata is not None:
        print('---number of successfully imported events---')
        print(import_operation.metadata.success_count)
        print('---number of failures during the importing---')
        print(import_operation.metadata.failure_count)
    else:
        print('---operation.metadata is empty---')
    if import_operation.result is not None:
        print('---operation result:---')
        print(import_operation.result())
    else:
        print('---operation.result is empty---')
import_user_events_from_inline_source()