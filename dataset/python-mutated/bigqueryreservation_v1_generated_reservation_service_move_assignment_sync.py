from google.cloud import bigquery_reservation_v1

def sample_move_assignment():
    if False:
        while True:
            i = 10
    client = bigquery_reservation_v1.ReservationServiceClient()
    request = bigquery_reservation_v1.MoveAssignmentRequest(name='name_value')
    response = client.move_assignment(request=request)
    print(response)