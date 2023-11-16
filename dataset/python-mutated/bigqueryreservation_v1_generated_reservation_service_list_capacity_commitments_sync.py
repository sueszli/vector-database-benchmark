from google.cloud import bigquery_reservation_v1

def sample_list_capacity_commitments():
    if False:
        while True:
            i = 10
    client = bigquery_reservation_v1.ReservationServiceClient()
    request = bigquery_reservation_v1.ListCapacityCommitmentsRequest(parent='parent_value')
    page_result = client.list_capacity_commitments(request=request)
    for response in page_result:
        print(response)