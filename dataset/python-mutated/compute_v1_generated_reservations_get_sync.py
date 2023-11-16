from google.cloud import compute_v1

def sample_get():
    if False:
        return 10
    client = compute_v1.ReservationsClient()
    request = compute_v1.GetReservationRequest(project='project_value', reservation='reservation_value', zone='zone_value')
    response = client.get(request=request)
    print(response)