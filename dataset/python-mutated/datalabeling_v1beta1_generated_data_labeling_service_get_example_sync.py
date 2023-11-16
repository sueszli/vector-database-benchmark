from google.cloud import datalabeling_v1beta1

def sample_get_example():
    if False:
        i = 10
        return i + 15
    client = datalabeling_v1beta1.DataLabelingServiceClient()
    request = datalabeling_v1beta1.GetExampleRequest(name='name_value')
    response = client.get_example(request=request)
    print(response)