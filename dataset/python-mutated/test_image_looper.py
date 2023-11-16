"""This executable test the pub/sub topic created by gcs_image_looper.py"""
from concurrent.futures import TimeoutError
from google.cloud import pubsub_v1
from google.api_core.exceptions import AlreadyExists
project_id = 'apache-beam-testing'
subscription_id = 'test-image-looper'
topic_id = 'Imagenet_openimage_50k_benchmark'
publisher = pubsub_v1.PublisherClient()
subscriber = pubsub_v1.SubscriberClient()
topic_path = publisher.topic_path(project_id, topic_id)
subscription_path = subscriber.subscription_path(project_id, subscription_id)
try:
    subscription = subscriber.create_subscription(request={'name': subscription_path, 'topic': topic_path})
    print(f'Subscription created: {subscription}')
except AlreadyExists:
    subscriber.delete_subscription(request={'subscription': subscription_path})
    subscription = subscriber.create_subscription(request={'name': subscription_path, 'topic': topic_path})
    print(f'Subscription recreated: {subscription}')
timeout = 3.0
total_images = []

def callback(message: pubsub_v1.subscriber.message.Message) -> None:
    if False:
        for i in range(10):
            print('nop')
    total_images.append(message.data.decode())
    message.ack()
streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
print(f'Listening for messages on {subscription_path}..\n')
try:
    streaming_pull_future.result(timeout=timeout)
except TimeoutError:
    streaming_pull_future.cancel()
    streaming_pull_future.result()
print('Results: \n', total_images)
subscriber.delete_subscription(request={'subscription': subscription_path})
print(f'Subscription deleted: {subscription_path}.')