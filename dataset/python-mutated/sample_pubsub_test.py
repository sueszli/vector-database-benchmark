import base64
from unittest import mock
import main
mock_context = mock.Mock()
mock_context.event_id = '617187464135194'
mock_context.timestamp = '2019-07-15T22:09:03.761Z'
mock_context.resource = {'name': 'projects/my-project/topics/my-topic', 'service': 'pubsub.googleapis.com', 'type': 'type.googleapis.com/google.pubsub.v1.PubsubMessage'}

def test_print_hello_world(capsys):
    if False:
        i = 10
        return i + 15
    data = {}
    main.hello_pubsub(data, mock_context)
    (out, err) = capsys.readouterr()
    assert 'Hello World!' in out

def test_print_name(capsys):
    if False:
        for i in range(10):
            print('nop')
    name = 'test'
    data = {'data': base64.b64encode(name.encode())}
    main.hello_pubsub(data, mock_context)
    (out, err) = capsys.readouterr()
    assert f'Hello {name}!\n' in out