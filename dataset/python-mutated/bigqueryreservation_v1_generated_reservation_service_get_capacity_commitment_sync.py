from google.cloud import bigquery_reservation_v1

def sample_get_capacity_commitment():
    if False:
        while True:
            i = 10
    client = bigquery_reservation_v1.ReservationServiceClient()
    request = bigquery_reservation_v1.GetCapacityCommitmentRequest(name='name_value')
    response = client.get_capacity_commitment(request=request)
    print(response)