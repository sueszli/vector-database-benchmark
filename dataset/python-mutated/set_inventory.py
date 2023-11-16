import asyncio
import random
import string
import time
from google.api_core.exceptions import GoogleAPICallError
import google.auth
from google.cloud.retail import FulfillmentInfo, PriceInfo, Product, ProductServiceClient, SetInventoryRequest
from google.protobuf.field_mask_pb2 import FieldMask
from setup_product.setup_cleanup import create_product, delete_product, get_product
project_id = google.auth.default()[1]
product_id = ''.join(random.sample(string.ascii_lowercase, 8))
product_name = 'projects/' + project_id + '/locations/global/catalogs/default_catalog/branches/default_branch/products/' + product_id

def get_product_with_inventory_info(product_name: str) -> Product:
    if False:
        i = 10
        return i + 15
    price_info = PriceInfo()
    price_info.price = 15.0
    price_info.original_price = 60.0
    price_info.cost = 8.0
    price_info.currency_code = 'USD'
    fulfillment_info = FulfillmentInfo()
    fulfillment_info.type_ = 'pickup-in-store'
    fulfillment_info.place_ids = ['store1', 'store2']
    product = Product()
    product.name = product_name
    product.price_info = price_info
    product.fulfillment_info = [fulfillment_info]
    product.availability = 'IN_STOCK'
    return product

def get_set_inventory_request(product_name: str) -> SetInventoryRequest:
    if False:
        return 10
    set_mask = FieldMask(paths=['price_info', 'availability', 'fulfillment_info', 'available_quantity'])
    set_inventory_request = SetInventoryRequest()
    set_inventory_request.inventory = get_product_with_inventory_info(product_name)
    set_inventory_request.allow_missing = True
    set_inventory_request.set_mask = set_mask
    print('---set inventory request---')
    print(set_inventory_request)
    return set_inventory_request

def set_inventory(product_name: str):
    if False:
        print('Hello World!')
    set_inventory_request = get_set_inventory_request(product_name)
    return ProductServiceClient().set_inventory(set_inventory_request)

async def set_inventory_and_remove_product(product_name: str):
    operation = set_inventory(product_name)
    try:
        while not operation.done():
            print('---please wait till operation is done---')
            time.sleep(30)
        print('---set inventory operation is done---')
    except GoogleAPICallError:
        pass
    get_product(product_name)
    delete_product(product_name)
product = create_product(product_id)
asyncio.run(set_inventory_and_remove_product(product.name))