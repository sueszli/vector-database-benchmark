from google.cloud import bigquery_reservation_v1

def sample_list_reservations():
    if False:
        return 10
    client = bigquery_reservation_v1.ReservationServiceClient()
    request = bigquery_reservation_v1.ListReservationsRequest(parent='parent_value')
    page_result = client.list_reservations(request=request)
    for response in page_result:
        print(response)