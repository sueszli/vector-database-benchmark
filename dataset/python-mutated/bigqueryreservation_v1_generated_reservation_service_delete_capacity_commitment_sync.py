from google.cloud import bigquery_reservation_v1

def sample_delete_capacity_commitment():
    if False:
        for i in range(10):
            print('nop')
    client = bigquery_reservation_v1.ReservationServiceClient()
    request = bigquery_reservation_v1.DeleteCapacityCommitmentRequest(name='name_value')
    client.delete_capacity_commitment(request=request)