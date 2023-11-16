import asyncio
import random
import string
import time
from google.api_core.exceptions import GoogleAPICallError
import google.auth
from google.cloud.retail import AddFulfillmentPlacesRequest, ProductServiceClient
from setup_product.setup_cleanup import create_product, delete_product, get_product
project_id = google.auth.default()[1]
product_id = ''.join(random.sample(string.ascii_lowercase, 8))
product_name = 'projects/' + project_id + '/locations/global/catalogs/default_catalog/branches/default_branch/products/' + product_id

def get_add_fulfillment_request(product_name: str, place_id) -> AddFulfillmentPlacesRequest:
    if False:
        print('Hello World!')
    add_fulfillment_request = AddFulfillmentPlacesRequest()
    add_fulfillment_request.product = product_name
    add_fulfillment_request.type_ = 'pickup-in-store'
    add_fulfillment_request.place_ids = [place_id]
    add_fulfillment_request.allow_missing = True
    print('---add fulfillment request---')
    print(add_fulfillment_request)
    return add_fulfillment_request

async def add_places(product_name: str):
    print('------add fulfillment places-----')
    add_fulfillment_request = get_add_fulfillment_request(product_name, 'store2')
    operation = ProductServiceClient().add_fulfillment_places(add_fulfillment_request)
    try:
        while not operation.done():
            print('---please wait till operation is done---')
            time.sleep(30)
        print('---add fulfillment places operation is done---')
    except GoogleAPICallError:
        pass
    get_product(product_name)
    delete_product(product_name)
create_product(product_id)
asyncio.run(add_places(product_name))