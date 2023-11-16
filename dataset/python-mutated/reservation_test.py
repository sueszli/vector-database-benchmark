import google.api_core.exceptions
from google.cloud.bigquery_reservation_v1.services import reservation_service
import pytest
import test_utils.prefixer
from . import reservation_create, reservation_delete
reservation_prefixer = test_utils.prefixer.Prefixer('py-bq-r', 'snippets', separator='-')

@pytest.fixture(scope='module', autouse=True)
def cleanup_reservations(reservation_client: reservation_service.ReservationServiceClient, location_path: str) -> None:
    if False:
        i = 10
        return i + 15
    for reservation in reservation_client.list_reservations(parent=location_path):
        reservation_id = reservation.name.split('/')[-1]
        if reservation_prefixer.should_cleanup(reservation_id):
            reservation_client.delete_reservation(name=reservation.name)

@pytest.fixture(scope='session')
def reservation_id(reservation_client: reservation_service.ReservationServiceClient, project_id: str, location: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    id_ = reservation_prefixer.create_prefix()
    yield id_
    reservation_name = reservation_client.reservation_path(project_id, location, id_)
    try:
        reservation_client.delete_reservation(name=reservation_name)
    except google.api_core.exceptions.NotFound:
        pass

@pytest.mark.parametrize('transport', ['grpc', 'rest'])
def test_reservation_samples(capsys: pytest.CaptureFixture, project_id: str, location: str, reservation_id: str, transport: str) -> None:
    if False:
        print('Hello World!')
    slot_capacity = 100
    reservation = reservation_create.create_reservation(project_id, location, reservation_id, slot_capacity, transport)
    assert reservation.slot_capacity == 100
    assert reservation_id in reservation.name
    (out, _) = capsys.readouterr()
    assert f'Created reservation: {reservation.name}' in out
    reservation_delete.delete_reservation(project_id, location, reservation_id, transport)
    (out, _) = capsys.readouterr()
    assert 'Deleted reservation' in out
    assert reservation_id in out