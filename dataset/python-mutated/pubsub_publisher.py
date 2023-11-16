""" This Cloud Function example creates Pub/Sub messages.

    Usage: Replace <PROJECT_ID> with the project ID of your project.
"""
from google.cloud import pubsub_v1
project = '<PROJECT_ID>'
topic = 'dag-topic-trigger'

def pubsub_publisher(request):
    if False:
        return 10
    'Publish message from HTTP request to Pub/Sub topic.\n    Args:\n        request (flask.Request): HTTP request object.\n    Returns:\n        The response text with message published into Pub/Sub topic\n        Response object using\n        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.\n    '
    request_json = request.get_json()
    print(request_json)
    if request.args and 'message' in request.args:
        data_str = request.args.get('message')
    elif request_json and 'message' in request_json:
        data_str = request_json['message']
    else:
        return "Message content not found! Use 'message' key to specify"
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project, topic)
    data = data_str.encode('utf-8')
    message_length = len(data_str)
    future = publisher.publish(topic_path, data, message_length=str(message_length))
    print(future.result())
    return f'Message {data} with message_length {message_length} published to {topic_path}.'