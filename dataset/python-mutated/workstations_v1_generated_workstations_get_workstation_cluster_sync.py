from google.cloud import workstations_v1

def sample_get_workstation_cluster():
    if False:
        i = 10
        return i + 15
    client = workstations_v1.WorkstationsClient()
    request = workstations_v1.GetWorkstationClusterRequest(name='name_value')
    response = client.get_workstation_cluster(request=request)
    print(response)