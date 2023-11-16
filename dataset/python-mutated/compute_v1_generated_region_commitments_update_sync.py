from google.cloud import compute_v1

def sample_update():
    if False:
        return 10
    client = compute_v1.RegionCommitmentsClient()
    request = compute_v1.UpdateRegionCommitmentRequest(commitment='commitment_value', project='project_value', region='region_value')
    response = client.update(request=request)
    print(response)