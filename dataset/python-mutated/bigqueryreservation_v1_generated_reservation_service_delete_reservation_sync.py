from google.cloud import bigquery_reservation_v1

def sample_delete_reservation():
    if False:
        while True:
            i = 10
    client = bigquery_reservation_v1.ReservationServiceClient()
    request = bigquery_reservation_v1.DeleteReservationRequest(name='name_value')
    client.delete_reservation(request=request)