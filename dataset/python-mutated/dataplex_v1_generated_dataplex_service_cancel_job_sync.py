from google.cloud import dataplex_v1

def sample_cancel_job():
    if False:
        for i in range(10):
            print('nop')
    client = dataplex_v1.DataplexServiceClient()
    request = dataplex_v1.CancelJobRequest(name='name_value')
    client.cancel_job(request=request)