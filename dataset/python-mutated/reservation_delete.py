def delete_reservation(project_id: str, location: str, reservation_id: str, transport: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    original_project_id = project_id
    original_location = location
    original_reservation_id = reservation_id
    original_transport = transport
    project_id = 'your-project-id'
    location = 'US'
    reservation_id = 'sample-reservation'
    transport = 'grpc'
    project_id = original_project_id
    location = original_location
    reservation_id = original_reservation_id
    transport = original_transport
    from google.cloud.bigquery_reservation_v1.services import reservation_service
    reservation_client = reservation_service.ReservationServiceClient(transport=transport)
    reservation_name = reservation_client.reservation_path(project_id, location, reservation_id)
    reservation_client.delete_reservation(name=reservation_name)
    print(f'Deleted reservation: {reservation_name}')