from google.cloud import compute_v1

def sample_set_iam_policy():
    if False:
        print('Hello World!')
    client = compute_v1.ReservationsClient()
    request = compute_v1.SetIamPolicyReservationRequest(project='project_value', resource='resource_value', zone='zone_value')
    response = client.set_iam_policy(request=request)
    print(response)