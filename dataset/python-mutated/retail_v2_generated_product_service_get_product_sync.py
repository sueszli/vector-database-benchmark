from google.cloud import retail_v2

def sample_get_product():
    if False:
        print('Hello World!')
    client = retail_v2.ProductServiceClient()
    request = retail_v2.GetProductRequest(name='name_value')
    response = client.get_product(request=request)
    print(response)