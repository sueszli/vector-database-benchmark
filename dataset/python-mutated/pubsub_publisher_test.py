from unittest import mock
import pytest
import pubsub_publisher

@pytest.fixture()
def dump_request_args():
    if False:
        while True:
            i = 10

    class Request:
        args = {'message': 'test with args'}

        def get_json(self):
            if False:
                i = 10
                return i + 15
            return self.args
    return Request()

@pytest.fixture()
def dump_request():
    if False:
        return 10

    class Request:
        args = None

        def get_json(self):
            if False:
                i = 10
                return i + 15
            return {'message': 'test with no args'}
    return Request()

@pytest.fixture()
def dump_request_no_message():
    if False:
        for i in range(10):
            print('nop')

    class Request:
        args = None

        def get_json(self):
            if False:
                for i in range(10):
                    print('nop')
            return {'no_message': 'test with no message key'}
    return Request()

def test_request_with_none():
    if False:
        i = 10
        return i + 15
    request = None
    with pytest.raises(Exception):
        pubsub_publisher.pubsub_publisher(request)

def test_content_not_found(dump_request_no_message):
    if False:
        return 10
    output = "Message content not found! Use 'message' key to specify"
    assert pubsub_publisher.pubsub_publisher(dump_request_no_message) == output, f"The function didn't return '{output}'"

@mock.patch('pubsub_publisher.pubsub_v1.PublisherClient.publish')
@mock.patch('pubsub_publisher.pubsub_v1.PublisherClient.topic_path')
def test_topic_path_args(topic_path, _, dump_request_args):
    if False:
        i = 10
        return i + 15
    pubsub_publisher.pubsub_publisher(dump_request_args)
    topic_path.assert_called_once_with('<PROJECT_ID>', 'dag-topic-trigger')

@mock.patch('pubsub_publisher.pubsub_v1.PublisherClient.publish')
def test_publish_args(publish, dump_request_args):
    if False:
        print('Hello World!')
    pubsub_publisher.pubsub_publisher(dump_request_args)
    publish.assert_called_once_with('projects/<PROJECT_ID>/topics/dag-topic-trigger', dump_request_args.args.get('message').encode('utf-8'), message_length=str(len(dump_request_args.args.get('message'))))

@mock.patch('pubsub_publisher.pubsub_v1.PublisherClient.publish')
@mock.patch('pubsub_publisher.pubsub_v1.PublisherClient.topic_path')
def test_topic_path(topic_path, _, dump_request):
    if False:
        for i in range(10):
            print('nop')
    pubsub_publisher.pubsub_publisher(dump_request)
    topic_path.assert_called_once_with('<PROJECT_ID>', 'dag-topic-trigger')

@mock.patch('pubsub_publisher.pubsub_v1.PublisherClient.publish')
def test_publish(publish, dump_request):
    if False:
        return 10
    pubsub_publisher.pubsub_publisher(dump_request)
    publish.assert_called_once_with('projects/<PROJECT_ID>/topics/dag-topic-trigger', dump_request.get_json().get('message').encode('utf-8'), message_length=str(len(dump_request.get_json().get('message'))))