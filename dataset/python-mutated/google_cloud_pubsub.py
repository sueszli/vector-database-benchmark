import os
from dataclasses import dataclass
from typing import Callable
from google.api_core import retry
from google.cloud import pubsub_v1
from google.oauth2 import service_account
from mage_ai.shared.config import BaseConfig
from mage_ai.streaming.constants import DEFAULT_BATCH_SIZE
from mage_ai.streaming.sources.base import BaseSource

@dataclass
class GoogleCloudPubSubConfig(BaseConfig):
    project_id: str
    topic_id: str
    subscription_id: str
    timeout: int = 5
    batch_size: int = DEFAULT_BATCH_SIZE
    pubsub_emulator_host: str = None
    path_to_credentials_json_file: str = None

class GoogleCloudPubSubSource(BaseSource):
    """
    Handles data transfer between a Google Cloud Pub/Sub topic and the Mage app.

    GOOGLE_APPLICATION_CREDENTIALS environment could be used to set the Google Cloud
    credentials file for authentication.
    """
    config_class = GoogleCloudPubSubConfig

    def _get_publisher_client(self) -> pubsub_v1.PublisherClient:
        if False:
            for i in range(10):
                print('nop')
        if self.config.path_to_credentials_json_file is not None:
            credentials = service_account.Credentials.from_service_account_file(self.config.path_to_credentials_json_file)
            return pubsub_v1.PublisherClient(credentials=credentials)
        else:
            return pubsub_v1.PublisherClient()

    def _get_subscriber_client(self) -> pubsub_v1.SubscriberClient:
        if False:
            print('Hello World!')
        if self.config.path_to_credentials_json_file is not None:
            credentials = service_account.Credentials.from_service_account_file(self.config.path_to_credentials_json_file)
            return pubsub_v1.SubscriberClient(credentials=credentials)
        else:
            return pubsub_v1.SubscriberClient()

    def _exist_subscription(self, project_id: str) -> bool:
        if False:
            return 10
        project_path = f'projects/{project_id}'
        subscriptions = self.subscriber_client.list_subscriptions(request={'project': project_path})
        for subscription in subscriptions:
            if subscription.name == self.subscription_path:
                return True
        return False

    def _create_subscription(self, project_id: str, topic_id: str, subscription_id: str) -> None:
        if False:
            i = 10
            return i + 15
        'Create a new pull subscription on the given topic.'
        if self._exist_subscription(project_id):
            self._print(f'Subscription already exists: {self.subscription_path}')
            return
        publisher = self._get_publisher_client()
        topic_path = publisher.topic_path(project_id, topic_id)
        subscription = self.subscriber_client.create_subscription(request={'name': self.subscription_path, 'topic': topic_path})
        self._print(f'Subscription created: {subscription}')

    def init_client(self) -> None:
        if False:
            print('Hello World!')
        if self.config.pubsub_emulator_host is not None:
            os.environ['PUBSUB_EMULATOR_HOST'] = self.config.pubsub_emulator_host
        self.subscriber_client = self._get_subscriber_client()
        self.subscription_path = self.subscriber_client.subscription_path(self.config.project_id, self.config.subscription_id)
        self._create_subscription(self.config.project_id, self.config.topic_id, self.config.subscription_id)

    def read(self, handler: Callable) -> None:
        if False:
            print('Hello World!')
        self._print('Start consuming messages.')

        def callback(received_message: pubsub_v1.subscriber.message.Message) -> None:
            if False:
                i = 10
                return i + 15
            handler(dict(data=received_message.message.data.decode(), metadata=dict(attributes=received_message.message.attributes)))
            received_message.ack()
        with self.subscriber_client:
            streaming_pull_future = self.subscriber_client.subscribe(self.subscription_path, callback=callback)
            try:
                self._print('Start receiving message with timeout: {self.config.timeout}')
                streaming_pull_future.result(timeout=self.config.timeout)
            except TimeoutError:
                streaming_pull_future.cancel()
                streaming_pull_future.result()

    def batch_read(self, handler: Callable) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._print('Start consuming batch messages.')
        if self.config.batch_size > 0:
            batch_size = self.config.batch_size
        else:
            batch_size = DEFAULT_BATCH_SIZE
        with self.subscriber_client:
            while True:
                response = self.subscriber_client.pull(request={'subscription': self.subscription_path, 'max_messages': batch_size}, retry=retry.Retry(deadline=300))
                if len(response.received_messages) == 0:
                    continue
                ack_ids = []
                message_values = []
                self._print(f'Number of received messages: {len(response.received_messages)}')
                for received_message in response.received_messages:
                    message_values.append(dict(data=received_message.message.data.decode(), metadata=dict(attributes=received_message.message.attributes)))
                    ack_ids.append(received_message.ack_id)
                handler(message_values)
                self.subscriber_client.acknowledge(request={'subscription': self.subscription_path, 'ack_ids': ack_ids})
                self._print(f'Received and acknowledged {len(response.received_messages)} messages from {self.subscription_path}.')