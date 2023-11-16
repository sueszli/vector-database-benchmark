from google.cloud import datalabeling_v1beta1

def sample_delete_instruction():
    if False:
        for i in range(10):
            print('nop')
    client = datalabeling_v1beta1.DataLabelingServiceClient()
    request = datalabeling_v1beta1.DeleteInstructionRequest(name='name_value')
    client.delete_instruction(request=request)