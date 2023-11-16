import os
from datetime import datetime
from azure.eventhub import EventHubConsumerClient, EventHubProducerClient, EventData
RECEIVE_TIMEOUT = 30
CONSUMER_GROUP = '$Default'
STARTING_POSITION = '-1'
TEST_EVENTS = [EventData(b'Test Event 1 in Python'), EventData(b'Test Event 2 in Python'), EventData(b'Test Event 3 in Python')]

class EventHub:

    def __init__(self):
        if False:
            print('Hello World!')
        connection_string = os.environ['EVENT_HUBS_CONNECTION_STRING']
        event_hub_name = 'myeventhub'
        self.consumer_client = EventHubConsumerClient.from_connection_string(connection_string, CONSUMER_GROUP, idle_timeout=RECEIVE_TIMEOUT)
        self.producer_client = EventHubProducerClient.from_connection_string(connection_string)
        self.received_event_count = 0

    def get_partition_ids(self):
        if False:
            print('Hello World!')
        print('Getting partitions id...')
        partition_ids = self.consumer_client.get_partition_ids()
        print('\tdone')
        return partition_ids

    def send_and_receive_events(self, partition_id):
        if False:
            i = 10
            return i + 15
        print('Sending events...')
        batch = self.producer_client.create_batch(partition_id=partition_id)
        for event in TEST_EVENTS:
            batch.add(event)
        self.producer_client.send_batch(batch)
        self.producer_client.close()
        print('\tdone')
        print('Receiving events...')
        self.consumer_client.receive(partition_id=partition_id, on_event=self.on_event, on_error=self.on_error, starting_position=STARTING_POSITION, starting_position_inclusive=True)
        print('\tdone')
        if self.received_event_count < len(TEST_EVENTS):
            raise Exception('Error, expecting {0} events, but {1} were received.'.format(str(len(TEST_EVENTS)), str(self.received_event_count)))

    def on_event(self, context, event):
        if False:
            return 10
        self.received_event_count += 1
        print(event.body_as_str())
        if self.received_event_count >= len(TEST_EVENTS):
            self.close_client()

    def on_error(self, context, error):
        if False:
            while True:
                i = 10
        self.consumer_client.close()
        raise Exception('Received Error: {0}'.format(error))

    def close_client(self):
        if False:
            while True:
                i = 10
        self.consumer_client.close()

    def run(self):
        if False:
            print('Hello World!')
        print('')
        print('------------------------')
        print('Event Hubs')
        print('------------------------')
        print('1) Get partition ID')
        print('2) Send Events')
        print('3) Consume Events')
        print('')
        partition_ids = self.get_partition_ids()
        self.send_and_receive_events(partition_ids[0])