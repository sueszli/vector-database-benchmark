from google.cloud import confidentialcomputing_v1

def sample_verify_attestation():
    if False:
        while True:
            i = 10
    client = confidentialcomputing_v1.ConfidentialComputingClient()
    request = confidentialcomputing_v1.VerifyAttestationRequest(challenge='challenge_value')
    response = client.verify_attestation(request=request)
    print(response)