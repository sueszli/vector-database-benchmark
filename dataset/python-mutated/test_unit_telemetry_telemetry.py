import json
from unittest import mock
import responses
from telemetry.telemetry import SelfHostedTelemetryWrapper
from tests.unit.telemetry.helpers import get_example_telemetry_data

@responses.activate
@mock.patch('telemetry.telemetry.TelemetryData')
def test_self_hosted_telemetry_wrapper_send_heartbeat(MockTelemetryData):
    if False:
        for i in range(10):
            print('nop')
    responses.add(responses.POST, SelfHostedTelemetryWrapper.TELEMETRY_API_URI, json={}, status=200)
    data = get_example_telemetry_data()
    mock_telemetry_data = mock.MagicMock(**data)
    MockTelemetryData.generate_telemetry_data.return_value = mock_telemetry_data
    SelfHostedTelemetryWrapper().send_heartbeat()
    assert len(responses.calls) == 1
    assert responses.calls[0].request.url == SelfHostedTelemetryWrapper.TELEMETRY_API_URI
    assert responses.calls[0].request.body.decode('utf-8') == json.dumps(data)