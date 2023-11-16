import random
import string
import google.auth
from google.cloud.retail import PriceInfo, Product, ProductServiceClient, UpdateProductRequest
from google.cloud.retail_v2.types import product
from setup_product.setup_cleanup import create_product, delete_product
project_id = google.auth.default()[1]
default_branch_name = 'projects/' + project_id + '/locations/global/catalogs/default_catalog/branches/default_branch'
generated_product_id = ''.join(random.sample(string.ascii_lowercase, 8))

def generate_product_for_update(product_id: str) -> Product:
    if False:
        while True:
            i = 10
    price_info = PriceInfo()
    price_info.price = 20.0
    price_info.original_price = 25.5
    price_info.currency_code = 'EUR'
    return product.Product(id=product_id, name='projects/' + project_id + '/locations/global/catalogs/default_catalog/branches/default_branch/products/' + product_id, title='Updated Nest Mini', type_=product.Product.Type.PRIMARY, categories=['Updated Speakers and displays'], brands=['Updated Google'], availability='OUT_OF_STOCK', price_info=price_info)

def get_update_product_request(product_to_update: Product):
    if False:
        i = 10
        return i + 15
    update_product_request = UpdateProductRequest()
    update_product_request.product = product_to_update
    update_product_request.allow_missing = True
    print('---update product request---')
    print(update_product_request)
    return update_product_request

def update_product(original_product: Product):
    if False:
        i = 10
        return i + 15
    updated_product = ProductServiceClient().update_product(get_update_product_request(generate_product_for_update(original_product.id)))
    print('---updated product---:')
    print(updated_product)
    return updated_product
created_product = create_product(generated_product_id)
update_product(created_product)
delete_product(created_product.name)