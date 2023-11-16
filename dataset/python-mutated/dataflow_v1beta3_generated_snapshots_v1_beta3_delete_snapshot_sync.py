from google.cloud import dataflow_v1beta3

def sample_delete_snapshot():
    if False:
        return 10
    client = dataflow_v1beta3.SnapshotsV1Beta3Client()
    request = dataflow_v1beta3.DeleteSnapshotRequest()
    response = client.delete_snapshot(request=request)
    print(response)