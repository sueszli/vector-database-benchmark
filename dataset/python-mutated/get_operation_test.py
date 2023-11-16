import google.auth
from google.cloud import optimization_v1
import pytest
import get_operation

@pytest.fixture(scope='function')
def operation_id() -> str:
    if False:
        for i in range(10):
            print('nop')
    fleet_routing_client = optimization_v1.FleetRoutingClient()
    (_, project_id) = google.auth.default()
    fleet_routing_request = {'parent': f'projects/{project_id}'}
    operation = fleet_routing_client.batch_optimize_tours(fleet_routing_request)
    yield operation.operation.name

def test_get_operation_status(capsys: pytest.LogCaptureFixture, operation_id: str) -> None:
    if False:
        print('Hello World!')
    get_operation.get_operation(operation_id)
    (out, _) = capsys.readouterr()
    assert 'Operation details' in out