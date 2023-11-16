from google.cloud import documentai_v1beta3

def sample_get_evaluation():
    if False:
        i = 10
        return i + 15
    client = documentai_v1beta3.DocumentProcessorServiceClient()
    request = documentai_v1beta3.GetEvaluationRequest(name='name_value')
    response = client.get_evaluation(request=request)
    print(response)