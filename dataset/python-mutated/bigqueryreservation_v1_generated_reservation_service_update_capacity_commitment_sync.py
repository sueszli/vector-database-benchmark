from google.cloud import bigquery_reservation_v1

def sample_update_capacity_commitment():
    if False:
        for i in range(10):
            print('nop')
    client = bigquery_reservation_v1.ReservationServiceClient()
    request = bigquery_reservation_v1.UpdateCapacityCommitmentRequest()
    response = client.update_capacity_commitment(request=request)
    print(response)