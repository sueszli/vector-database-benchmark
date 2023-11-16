from google.cloud import datastream_v1

def sample_stop_backfill_job():
    if False:
        print('Hello World!')
    client = datastream_v1.DatastreamClient()
    request = datastream_v1.StopBackfillJobRequest(object_='object__value')
    response = client.stop_backfill_job(request=request)
    print(response)