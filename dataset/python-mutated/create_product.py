import random
import string
import google.auth
from google.cloud.retail import CreateProductRequest, PriceInfo, Product, ProductServiceClient
from google.cloud.retail_v2.types import product
from setup_product.setup_cleanup import delete_product
project_id = google.auth.default()[1]
default_branch_name = 'projects/' + project_id + '/locations/global/catalogs/default_catalog/branches/default_branch'
generated_product_id = ''.join(random.sample(string.ascii_lowercase, 8))

def generate_product() -> Product:
    if False:
        i = 10
        return i + 15
    price_info = PriceInfo()
    price_info.price = 30.0
    price_info.original_price = 35.5
    price_info.currency_code = 'USD'
    return product.Product(title='Nest Mini', type_=product.Product.Type.PRIMARY, categories=['Speakers and displays'], brands=['Google'], price_info=price_info, availability='IN_STOCK')

def get_create_product_request(product_to_create: Product, product_id: str) -> object:
    if False:
        return 10
    create_product_request = CreateProductRequest()
    create_product_request.product = product_to_create
    create_product_request.product_id = product_id
    create_product_request.parent = default_branch_name
    print('---create product request---')
    print(create_product_request)
    return create_product_request

def create_product(product_id: str):
    if False:
        for i in range(10):
            print('nop')
    create_product_request = get_create_product_request(generate_product(), product_id)
    product_created = ProductServiceClient().create_product(create_product_request)
    print('---created product:---')
    print(product_created)
    return product_created
created_product = create_product(generated_product_id)
delete_product(created_product.name)