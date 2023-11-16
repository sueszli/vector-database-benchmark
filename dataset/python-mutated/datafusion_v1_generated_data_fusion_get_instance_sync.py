from google.cloud import data_fusion_v1

def sample_get_instance():
    if False:
        while True:
            i = 10
    client = data_fusion_v1.DataFusionClient()
    request = data_fusion_v1.GetInstanceRequest(name='name_value')
    response = client.get_instance(request=request)
    print(response)