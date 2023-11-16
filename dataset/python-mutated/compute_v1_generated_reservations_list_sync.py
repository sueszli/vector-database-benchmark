from google.cloud import compute_v1

def sample_list():
    if False:
        return 10
    client = compute_v1.ReservationsClient()
    request = compute_v1.ListReservationsRequest(project='project_value', zone='zone_value')
    page_result = client.list(request=request)
    for response in page_result:
        print(response)