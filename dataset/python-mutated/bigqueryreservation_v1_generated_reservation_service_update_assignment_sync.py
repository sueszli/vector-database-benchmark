from google.cloud import bigquery_reservation_v1

def sample_update_assignment():
    if False:
        print('Hello World!')
    client = bigquery_reservation_v1.ReservationServiceClient()
    request = bigquery_reservation_v1.UpdateAssignmentRequest()
    response = client.update_assignment(request=request)
    print(response)