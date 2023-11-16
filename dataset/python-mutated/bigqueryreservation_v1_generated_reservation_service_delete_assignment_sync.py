from google.cloud import bigquery_reservation_v1

def sample_delete_assignment():
    if False:
        i = 10
        return i + 15
    client = bigquery_reservation_v1.ReservationServiceClient()
    request = bigquery_reservation_v1.DeleteAssignmentRequest(name='name_value')
    client.delete_assignment(request=request)