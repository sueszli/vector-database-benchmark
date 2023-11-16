from google.cloud import bigquery_reservation_v1

def sample_create_assignment():
    if False:
        while True:
            i = 10
    client = bigquery_reservation_v1.ReservationServiceClient()
    request = bigquery_reservation_v1.CreateAssignmentRequest(parent='parent_value')
    response = client.create_assignment(request=request)
    print(response)