from google.cloud import compute_v1

def sample_update():
    if False:
        i = 10
        return i + 15
    client = compute_v1.ReservationsClient()
    request = compute_v1.UpdateReservationRequest(project='project_value', reservation='reservation_value', zone='zone_value')
    response = client.update(request=request)
    print(response)