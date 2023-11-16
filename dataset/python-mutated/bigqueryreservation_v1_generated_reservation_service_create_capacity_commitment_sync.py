from google.cloud import bigquery_reservation_v1

def sample_create_capacity_commitment():
    if False:
        return 10
    client = bigquery_reservation_v1.ReservationServiceClient()
    request = bigquery_reservation_v1.CreateCapacityCommitmentRequest(parent='parent_value')
    response = client.create_capacity_commitment(request=request)
    print(response)