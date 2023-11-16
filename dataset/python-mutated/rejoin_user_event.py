import google.auth
from google.cloud.retail import RejoinUserEventsRequest, UserEventServiceClient
from setup_events.setup_cleanup import purge_user_event, write_user_event
project_id = google.auth.default()[1]
default_catalog = f'projects/{project_id}/locations/global/catalogs/default_catalog'
visitor_id = 'test_visitor_id'

def get_rejoin_user_event_request():
    if False:
        while True:
            i = 10
    rejoin_user_event_request = RejoinUserEventsRequest()
    rejoin_user_event_request.parent = default_catalog
    rejoin_user_event_request.user_event_rejoin_scope = RejoinUserEventsRequest.UserEventRejoinScope.UNJOINED_EVENTS
    print('---rejoin user events request---')
    print(rejoin_user_event_request)
    return rejoin_user_event_request

def call_rejoin_user_events():
    if False:
        while True:
            i = 10
    rejoin_operation = UserEventServiceClient().rejoin_user_events(get_rejoin_user_event_request())
    print('---the rejoin operation was started:----')
    print(rejoin_operation.operation.name)
write_user_event(visitor_id)
call_rejoin_user_events()
purge_user_event(visitor_id)