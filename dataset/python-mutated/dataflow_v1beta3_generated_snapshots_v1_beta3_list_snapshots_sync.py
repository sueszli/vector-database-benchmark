from google.cloud import dataflow_v1beta3

def sample_list_snapshots():
    if False:
        return 10
    client = dataflow_v1beta3.SnapshotsV1Beta3Client()
    request = dataflow_v1beta3.ListSnapshotsRequest()
    response = client.list_snapshots(request=request)
    print(response)