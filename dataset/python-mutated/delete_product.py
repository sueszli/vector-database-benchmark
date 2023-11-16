import random
import string
import google.auth
from google.cloud.retail import DeleteProductRequest, ProductServiceClient
from setup_product.setup_cleanup import create_product
project_id = google.auth.default()[1]
default_branch_name = 'projects/' + project_id + '/locations/global/catalogs/default_catalog/branches/default_branch'
product_id = ''.join(random.sample(string.ascii_lowercase, 8))

def get_delete_product_request(product_name: str):
    if False:
        while True:
            i = 10
    delete_product_request = DeleteProductRequest()
    delete_product_request.name = product_name
    print('---delete product request---')
    print(delete_product_request)
    return delete_product_request

def delete_product(product_name: str):
    if False:
        i = 10
        return i + 15
    delete_product_request = get_delete_product_request(product_name)
    ProductServiceClient().delete_product(delete_product_request)
    print('deleting product ' + product_name)
    print('---product was deleted:---')
created_product_name = create_product(product_id).name
delete_product(created_product_name)