import datetime
import google.auth
from google.cloud.retail import UserEvent, UserEventServiceClient, WriteUserEventRequest
from google.protobuf.timestamp_pb2 import Timestamp
from setup_events.setup_cleanup import purge_user_event
project_id = google.auth.default()[1]
default_catalog = f'projects/{project_id}/locations/global/catalogs/default_catalog'
visitor_id = 'test_visitor_id'

def get_user_event():
    if False:
        print('Hello World!')
    timestamp = Timestamp()
    timestamp.seconds = int(datetime.datetime.now().timestamp())
    user_event = UserEvent()
    user_event.event_type = 'home-page-view'
    user_event.visitor_id = visitor_id
    user_event.event_time = timestamp
    print(user_event)
    return user_event

def get_write_event_request(user_event):
    if False:
        i = 10
        return i + 15
    write_user_event_request = WriteUserEventRequest()
    write_user_event_request.user_event = user_event
    write_user_event_request.parent = default_catalog
    print('---write user event request---')
    print(write_user_event_request)
    return write_user_event_request

def write_user_event():
    if False:
        return 10
    write_user_event_request = get_write_event_request(get_user_event())
    user_event = UserEventServiceClient().write_user_event(write_user_event_request)
    print('---written user event:---')
    print(user_event)
    return user_event
write_user_event()
purge_user_event(visitor_id)