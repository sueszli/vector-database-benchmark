from google.cloud import dataplex_v1

def sample_get_lake():
    if False:
        for i in range(10):
            print('nop')
    client = dataplex_v1.DataplexServiceClient()
    request = dataplex_v1.GetLakeRequest(name='name_value')
    response = client.get_lake(request=request)
    print(response)