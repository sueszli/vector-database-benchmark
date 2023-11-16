from google.cloud import datalabeling_v1beta1

def sample_get_instruction():
    if False:
        while True:
            i = 10
    client = datalabeling_v1beta1.DataLabelingServiceClient()
    request = datalabeling_v1beta1.GetInstructionRequest(name='name_value')
    response = client.get_instruction(request=request)
    print(response)