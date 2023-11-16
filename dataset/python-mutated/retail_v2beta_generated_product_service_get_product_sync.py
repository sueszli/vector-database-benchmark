from google.cloud import retail_v2beta

def sample_get_product():
    if False:
        while True:
            i = 10
    client = retail_v2beta.ProductServiceClient()
    request = retail_v2beta.GetProductRequest(name='name_value')
    response = client.get_product(request=request)
    print(response)