import google.auth
from google.cloud.retail import PurgeUserEventsRequest, UserEventServiceClient
from setup_events.setup_cleanup import write_user_event
project_id = google.auth.default()[1]
default_catalog = f'projects/{project_id}/locations/global/catalogs/default_catalog'
visitor_id = 'test_visitor_id'

def get_purge_user_event_request():
    if False:
        while True:
            i = 10
    purge_user_event_request = PurgeUserEventsRequest()
    purge_user_event_request.filter = f'visitorId="{visitor_id}"'
    purge_user_event_request.parent = default_catalog
    purge_user_event_request.force = True
    print('---purge user events request---')
    print(purge_user_event_request)
    return purge_user_event_request

def call_purge_user_events():
    if False:
        return 10
    purge_operation = UserEventServiceClient().purge_user_events(get_purge_user_event_request())
    print('---the purge operation was started:----')
    print(purge_operation.operation.name)
write_user_event(visitor_id)
call_purge_user_events()