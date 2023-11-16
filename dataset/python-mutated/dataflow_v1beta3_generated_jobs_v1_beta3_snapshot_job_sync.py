from google.cloud import dataflow_v1beta3

def sample_snapshot_job():
    if False:
        for i in range(10):
            print('nop')
    client = dataflow_v1beta3.JobsV1Beta3Client()
    request = dataflow_v1beta3.SnapshotJobRequest()
    response = client.snapshot_job(request=request)
    print(response)