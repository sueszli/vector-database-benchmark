from google.cloud import bigquery_reservation_v1

def sample_get_bi_reservation():
    if False:
        print('Hello World!')
    client = bigquery_reservation_v1.ReservationServiceClient()
    request = bigquery_reservation_v1.GetBiReservationRequest(name='name_value')
    response = client.get_bi_reservation(request=request)
    print(response)