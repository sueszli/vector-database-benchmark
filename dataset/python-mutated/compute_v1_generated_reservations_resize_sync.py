from google.cloud import compute_v1

def sample_resize():
    if False:
        print('Hello World!')
    client = compute_v1.ReservationsClient()
    request = compute_v1.ResizeReservationRequest(project='project_value', reservation='reservation_value', zone='zone_value')
    response = client.resize(request=request)
    print(response)