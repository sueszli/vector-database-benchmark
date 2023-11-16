from google.cloud import compute_v1

def sample_delete():
    if False:
        return 10
    client = compute_v1.ReservationsClient()
    request = compute_v1.DeleteReservationRequest(project='project_value', reservation='reservation_value', zone='zone_value')
    response = client.delete(request=request)
    print(response)