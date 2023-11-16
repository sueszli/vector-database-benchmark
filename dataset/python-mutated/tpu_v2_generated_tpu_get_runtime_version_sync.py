from google.cloud import tpu_v2

def sample_get_runtime_version():
    if False:
        while True:
            i = 10
    client = tpu_v2.TpuClient()
    request = tpu_v2.GetRuntimeVersionRequest(name='name_value')
    response = client.get_runtime_version(request=request)
    print(response)