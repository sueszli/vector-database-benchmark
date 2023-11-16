import random
import string
import google.auth
from google.cloud.retail import GetProductRequest, ProductServiceClient
from setup_product.setup_cleanup import create_product, delete_product
project_id = google.auth.default()[1]
product_id = ''.join(random.sample(string.ascii_lowercase, 8))

def get_product_request(product_name: str) -> object:
    if False:
        print('Hello World!')
    get_product_request = GetProductRequest()
    get_product_request.name = product_name
    print('---get product request---')
    print(get_product_request)
    return get_product_request

def get_product(product_name: str):
    if False:
        for i in range(10):
            print('nop')
    get_request = get_product_request(product_name)
    get_product_response = ProductServiceClient().get_product(get_request)
    print('---get product response:---')
    print(get_product_response)
    return get_product_response
created_product = create_product(product_id)
product = get_product(created_product.name)
delete_product(created_product.name)