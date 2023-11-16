import time
from google.api_core.exceptions import AlreadyExists
from google.cloud.pubsub import SubscriberClient
from google.cloud.pubsub_v1.subscriber.message import Message

def pubsub(subscription_id: str, timeout_seconds: int, project_id: str) -> int:
    if False:
        while True:
            i = 10
    'Respond to incoming occurrences using a Cloud Pub/Sub subscription.'
    client = SubscriberClient()
    subscription_name = client.subscription_path(project_id, subscription_id)
    receiver = MessageReceiver()
    client.subscribe(subscription_name, receiver.pubsub_callback)
    for _ in range(timeout_seconds):
        time.sleep(1)
    print(receiver.msg_count)
    return receiver.msg_count

class MessageReceiver:
    """Custom class to handle incoming Pub/Sub messages."""

    def __init__(self) -> None:
        if False:
            return 10
        self.msg_count = 0

    def pubsub_callback(self, message: Message) -> None:
        if False:
            i = 10
            return i + 15
        self.msg_count += 1
        print(f'Message {self.msg_count}: {message.data}')
        message.ack()

def create_occurrence_subscription(subscription_id: str, project_id: str) -> bool:
    if False:
        while True:
            i = 10
    'Creates a new Pub/Sub subscription object listening to the\n    Container Analysis Occurrences topic.'
    topic_id = 'container-analysis-occurrences-v1'
    client = SubscriberClient()
    topic_name = f'projects/{project_id}/topics/{topic_id}'
    subscription_name = client.subscription_path(project_id, subscription_id)
    success = True
    try:
        client.create_subscription({'name': subscription_name, 'topic': topic_name})
    except AlreadyExists:
        pass
    else:
        success = False
    return success