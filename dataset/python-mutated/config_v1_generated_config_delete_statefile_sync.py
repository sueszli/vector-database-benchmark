from google.cloud import config_v1

def sample_delete_statefile():
    if False:
        while True:
            i = 10
    client = config_v1.ConfigClient()
    request = config_v1.DeleteStatefileRequest(name='name_value', lock_id=725)
    client.delete_statefile(request=request)