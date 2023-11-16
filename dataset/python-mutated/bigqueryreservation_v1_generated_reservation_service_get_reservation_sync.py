from google.cloud import bigquery_reservation_v1

def sample_get_reservation():
    if False:
        for i in range(10):
            print('nop')
    client = bigquery_reservation_v1.ReservationServiceClient()
    request = bigquery_reservation_v1.GetReservationRequest(name='name_value')
    response = client.get_reservation(request=request)
    print(response)