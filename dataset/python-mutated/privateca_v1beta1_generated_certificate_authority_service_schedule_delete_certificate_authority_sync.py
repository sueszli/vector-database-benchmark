from google.cloud.security import privateca_v1beta1

def sample_schedule_delete_certificate_authority():
    if False:
        for i in range(10):
            print('nop')
    client = privateca_v1beta1.CertificateAuthorityServiceClient()
    request = privateca_v1beta1.ScheduleDeleteCertificateAuthorityRequest(name='name_value')
    operation = client.schedule_delete_certificate_authority(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)