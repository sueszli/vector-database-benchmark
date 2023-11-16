from google.cloud import bigquery_reservation_v1

def sample_merge_capacity_commitments():
    if False:
        i = 10
        return i + 15
    client = bigquery_reservation_v1.ReservationServiceClient()
    request = bigquery_reservation_v1.MergeCapacityCommitmentsRequest()
    response = client.merge_capacity_commitments(request=request)
    print(response)