from google.maps import addressvalidation_v1

def sample_validate_address():
    if False:
        for i in range(10):
            print('nop')
    client = addressvalidation_v1.AddressValidationClient()
    request = addressvalidation_v1.ValidateAddressRequest()
    response = client.validate_address(request=request)
    print(response)