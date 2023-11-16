from google.cloud import bigquery_reservation_v1

def sample_search_assignments():
    if False:
        print('Hello World!')
    client = bigquery_reservation_v1.ReservationServiceClient()
    request = bigquery_reservation_v1.SearchAssignmentsRequest(parent='parent_value')
    page_result = client.search_assignments(request=request)
    for response in page_result:
        print(response)