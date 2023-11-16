from google.cloud.bigquery_reservation_v1.types import reservation as reservation_types

def update_reservation(project_id: str, location: str, reservation_id: str, slot_capacity: str, transport: str) -> reservation_types.Reservation:
    if False:
        i = 10
        return i + 15
    original_project_id = project_id
    original_location = location
    original_reservation_id = reservation_id
    original_slot_capacity = slot_capacity
    original_transport = transport
    project_id = 'your-project-id'
    location = 'US'
    reservation_id = 'sample-reservation'
    slot_capacity = 50
    transport = 'grpc'
    project_id = original_project_id
    location = original_location
    reservation_id = original_reservation_id
    slot_capacity = original_slot_capacity
    transport = original_transport
    from google.cloud.bigquery_reservation_v1.services import reservation_service
    from google.cloud.bigquery_reservation_v1.types import reservation as reservation_types
    from google.protobuf import field_mask_pb2
    reservation_client = reservation_service.ReservationServiceClient(transport=transport)
    reservation_name = reservation_client.reservation_path(project_id, location, reservation_id)
    reservation = reservation_types.Reservation(name=reservation_name, slot_capacity=slot_capacity)
    field_mask = field_mask_pb2.FieldMask(paths=['slot_capacity'])
    reservation = reservation_client.update_reservation(reservation=reservation, update_mask=field_mask)
    print(f'Updated reservation: {reservation.name}')
    print(f'\tslot_capacity: {reservation.slot_capacity}')
    return reservation