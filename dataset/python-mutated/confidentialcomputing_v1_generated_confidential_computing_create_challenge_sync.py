from google.cloud import confidentialcomputing_v1

def sample_create_challenge():
    if False:
        while True:
            i = 10
    client = confidentialcomputing_v1.ConfidentialComputingClient()
    request = confidentialcomputing_v1.CreateChallengeRequest(parent='parent_value')
    response = client.create_challenge(request=request)
    print(response)