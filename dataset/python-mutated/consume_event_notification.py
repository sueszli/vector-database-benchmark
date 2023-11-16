"""
sample code for consuming an event notification in a cloud function.
"""
import base64

def consume_event_notification(event: dict, unused_context: None) -> str:
    if False:
        i = 10
        return i + 15
    '\n    consume_event_notification demonstrates how to consume and process a\n    Pub/Sub notification from Secret Manager.\n    Args:\n          event (dict): Event payload.\n          unused_context (google.cloud.functions.Context): Metadata for the event.\n    '
    event_type = event['attributes']['eventType']
    secret_id = event['attributes']['secretId']
    secret_metadata = base64.b64decode(event['data']).decode('utf-8')
    event_notification = f'Received {event_type} for {secret_id}. New metadata: {secret_metadata}'
    print(event_notification)
    return event_notification