from google.cloud import bigquery_reservation_v1

def sample_update_reservation():
    if False:
        for i in range(10):
            print('nop')
    client = bigquery_reservation_v1.ReservationServiceClient()
    request = bigquery_reservation_v1.UpdateReservationRequest()
    response = client.update_reservation(request=request)
    print(response)