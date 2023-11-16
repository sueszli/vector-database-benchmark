"""Tests for Eventlogging in JupyterHub.

To test a new schema or event, simply add it to the
`valid_events` and `invalid_events` variables below.

You *shouldn't* need to write new tests.
"""
import io
import json
import logging
from unittest import mock
import jsonschema
import pytest
from traitlets.config import Config
valid_events = [('hub.jupyter.org/server-action', 1, dict(action='start', username='test-username', servername='test-servername'))]
invalid_events = [('hub.jupyter.org/server-action', 1, dict(action='start'))]

@pytest.fixture
def eventlog_sink(app):
    if False:
        for i in range(10):
            print('nop')
    'Return eventlog and sink objects'
    sink = io.StringIO()
    handler = logging.StreamHandler(sink)
    cfg = Config()
    cfg.EventLog.handlers = [handler]
    with mock.patch.object(app.config, 'EventLog', cfg.EventLog):
        app.init_eventlog()
        yield (app.eventlog, sink)
    app.init_eventlog()

@pytest.mark.parametrize('schema, version, event', valid_events)
def test_valid_events(eventlog_sink, schema, version, event):
    if False:
        print('Hello World!')
    (eventlog, sink) = eventlog_sink
    eventlog.allowed_schemas = [schema]
    eventlog.record_event(schema, version, event)
    output = sink.getvalue()
    assert output
    data = json.loads(output)
    assert data is not None

@pytest.mark.parametrize('schema, version, event', invalid_events)
def test_invalid_events(eventlog_sink, schema, version, event):
    if False:
        i = 10
        return i + 15
    (eventlog, sink) = eventlog_sink
    eventlog.allowed_schemas = [schema]
    with pytest.raises(jsonschema.ValidationError):
        recorded_event = eventlog.record_event(schema, version, event)