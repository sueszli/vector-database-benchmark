from google.ai import generativelanguage_v1beta3

def sample_transfer_ownership():
    if False:
        while True:
            i = 10
    client = generativelanguage_v1beta3.PermissionServiceClient()
    request = generativelanguage_v1beta3.TransferOwnershipRequest(name='name_value', email_address='email_address_value')
    response = client.transfer_ownership(request=request)
    print(response)