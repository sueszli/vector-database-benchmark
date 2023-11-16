import random
import string
import google.auth
from google.cloud.retail import CreateProductRequest, DeleteProductRequest, GetProductRequest, PriceInfo, Product, ProductServiceClient, UpdateProductRequest
from google.cloud.retail_v2.types import product
project_id = google.auth.default()[1]
default_branch_name = 'projects/' + project_id + '/locations/global/catalogs/default_catalog/branches/default_branch'
product_id = ''.join(random.sample(string.ascii_lowercase, 8))
product_name = f'{default_branch_name}/products/{product_id}'

def generate_product() -> Product:
    if False:
        return 10
    price_info = PriceInfo()
    price_info.price = 30.0
    price_info.original_price = 35.5
    price_info.currency_code = 'USD'
    return product.Product(title='Nest Mini', type_=product.Product.Type.PRIMARY, categories=['Speakers and displays'], brands=['Google'], price_info=price_info, availability='IN_STOCK')

def generate_product_for_update() -> Product:
    if False:
        i = 10
        return i + 15
    price_info = PriceInfo()
    price_info.price = 20.0
    price_info.original_price = 25.5
    price_info.currency_code = 'EUR'
    return product.Product(id=product_id, name=product_name, title='Updated Nest Mini', type_=product.Product.Type.PRIMARY, categories=['Updated Speakers and displays'], brands=['Updated Google'], availability='OUT_OF_STOCK', price_info=price_info)

def create_product() -> object:
    if False:
        i = 10
        return i + 15
    create_product_request = CreateProductRequest()
    create_product_request.product = generate_product()
    create_product_request.product_id = product_id
    create_product_request.parent = default_branch_name
    print('---create product request---')
    print(create_product_request)
    product_created = ProductServiceClient().create_product(create_product_request)
    print('---created product:---')
    print(product_created)
    return product_created

def get_product() -> object:
    if False:
        for i in range(10):
            print('nop')
    get_product_request = GetProductRequest()
    get_product_request.name = product_name
    print('---get product request---')
    print(get_product_request)
    get_product_response = ProductServiceClient().get_product(get_product_request)
    print('---get product response:---')
    print(get_product_response)
    return get_product_response

def update_product():
    if False:
        i = 10
        return i + 15
    update_product_request = UpdateProductRequest()
    update_product_request.product = generate_product_for_update()
    update_product_request.allow_missing = True
    print('---update product request---')
    print(update_product_request)
    updated_product = ProductServiceClient().update_product(update_product_request)
    print('---updated product---:')
    print(updated_product)
    return updated_product

def delete_product():
    if False:
        for i in range(10):
            print('nop')
    delete_product_request = DeleteProductRequest()
    delete_product_request.name = product_name
    print('---delete product request---')
    print(delete_product_request)
    ProductServiceClient().delete_product(delete_product_request)
    print('deleting product ' + product_name)
    print('---product was deleted:---')
create_product()
get_product()
update_product()
delete_product()