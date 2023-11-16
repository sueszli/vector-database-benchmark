from google.cloud import bigquery_reservation_v1

def sample_update_bi_reservation():
    if False:
        print('Hello World!')
    client = bigquery_reservation_v1.ReservationServiceClient()
    request = bigquery_reservation_v1.UpdateBiReservationRequest()
    response = client.update_bi_reservation(request=request)
    print(response)