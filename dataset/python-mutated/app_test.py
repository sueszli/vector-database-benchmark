"""
A sample app demonstrating Google Cloud Trace
"""
import os
from unittest import mock
import app

def test_traces() -> None:
    if False:
        print('Hello World!')
    expected = 'Lorem ipsum dolor sit amet'
    os.environ['KEYWORD'] = expected
    app.app.testing = True
    exporter = mock.Mock()
    app.configure_exporter(exporter)
    client = app.app.test_client()
    resp = client.get('/')
    assert resp.status_code == 200
    assert expected in resp.data.decode('utf-8')