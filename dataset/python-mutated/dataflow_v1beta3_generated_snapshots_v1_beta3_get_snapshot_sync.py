from google.cloud import dataflow_v1beta3

def sample_get_snapshot():
    if False:
        while True:
            i = 10
    client = dataflow_v1beta3.SnapshotsV1Beta3Client()
    request = dataflow_v1beta3.GetSnapshotRequest()
    response = client.get_snapshot(request=request)
    print(response)