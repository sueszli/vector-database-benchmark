from google.ai import generativelanguage_v1beta3

def sample_delete_tuned_model():
    if False:
        return 10
    client = generativelanguage_v1beta3.ModelServiceClient()
    request = generativelanguage_v1beta3.DeleteTunedModelRequest(name='name_value')
    client.delete_tuned_model(request=request)