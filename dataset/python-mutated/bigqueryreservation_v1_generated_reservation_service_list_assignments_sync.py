from google.cloud import bigquery_reservation_v1

def sample_list_assignments():
    if False:
        i = 10
        return i + 15
    client = bigquery_reservation_v1.ReservationServiceClient()
    request = bigquery_reservation_v1.ListAssignmentsRequest(parent='parent_value')
    page_result = client.list_assignments(request=request)
    for response in page_result:
        print(response)