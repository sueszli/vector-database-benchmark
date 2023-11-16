from google.cloud import binaryauthorization_v1

def sample_validate_attestation_occurrence():
    if False:
        while True:
            i = 10
    client = binaryauthorization_v1.ValidationHelperV1Client()
    request = binaryauthorization_v1.ValidateAttestationOccurrenceRequest(attestor='attestor_value', occurrence_note='occurrence_note_value', occurrence_resource_uri='occurrence_resource_uri_value')
    response = client.validate_attestation_occurrence(request=request)
    print(response)