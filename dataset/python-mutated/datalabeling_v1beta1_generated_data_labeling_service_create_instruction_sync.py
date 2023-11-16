from google.cloud import datalabeling_v1beta1

def sample_create_instruction():
    if False:
        return 10
    client = datalabeling_v1beta1.DataLabelingServiceClient()
    request = datalabeling_v1beta1.CreateInstructionRequest(parent='parent_value')
    operation = client.create_instruction(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)