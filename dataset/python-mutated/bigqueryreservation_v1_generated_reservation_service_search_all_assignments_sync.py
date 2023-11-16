from google.cloud import bigquery_reservation_v1

def sample_search_all_assignments():
    if False:
        return 10
    client = bigquery_reservation_v1.ReservationServiceClient()
    request = bigquery_reservation_v1.SearchAllAssignmentsRequest(parent='parent_value')
    page_result = client.search_all_assignments(request=request)
    for response in page_result:
        print(response)