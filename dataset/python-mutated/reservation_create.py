from google.cloud.bigquery_reservation_v1.types import reservation as reservation_types

def create_reservation(project_id: str, location: str, reservation_id: str, slot_capacity: str, transport: str) -> reservation_types.Reservation:
    if False:
        while True:
            i = 10
    original_project_id = project_id
    original_location = location
    original_reservation_id = reservation_id
    original_slot_capacity = slot_capacity
    original_transport = transport
    project_id = 'your-project-id'
    location = 'US'
    reservation_id = 'sample-reservation'
    slot_capacity = 100
    transport = 'grpc'
    project_id = original_project_id
    location = original_location
    reservation_id = original_reservation_id
    slot_capacity = original_slot_capacity
    transport = original_transport
    from google.cloud.bigquery_reservation_v1.services import reservation_service
    from google.cloud.bigquery_reservation_v1.types import reservation as reservation_types
    reservation_client = reservation_service.ReservationServiceClient(transport=transport)
    parent = reservation_client.common_location_path(project_id, location)
    reservation = reservation_types.Reservation(slot_capacity=slot_capacity)
    reservation = reservation_client.create_reservation(parent=parent, reservation=reservation, reservation_id=reservation_id)
    print(f'Created reservation: {reservation.name}')
    return reservation