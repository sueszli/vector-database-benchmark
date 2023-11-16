from google.cloud import datastream_v1

def sample_start_backfill_job():
    if False:
        print('Hello World!')
    client = datastream_v1.DatastreamClient()
    request = datastream_v1.StartBackfillJobRequest(object_='object__value')
    response = client.start_backfill_job(request=request)
    print(response)