from google.cloud import bigquery_reservation_v1

def sample_create_reservation():
    if False:
        while True:
            i = 10
    client = bigquery_reservation_v1.ReservationServiceClient()
    request = bigquery_reservation_v1.CreateReservationRequest(parent='parent_value')
    response = client.create_reservation(request=request)
    print(response)