import asyncio
import random
import string
import time
from google.api_core.exceptions import GoogleAPICallError
from google.cloud.retail import ProductServiceClient, RemoveFulfillmentPlacesRequest
from setup_product.setup_cleanup import create_product, delete_product, get_product
product_id = ''.join(random.sample(string.ascii_lowercase, 8))

def get_remove_fulfillment_request(product_name: str, store_id) -> RemoveFulfillmentPlacesRequest:
    if False:
        return 10
    remove_fulfillment_request = RemoveFulfillmentPlacesRequest()
    remove_fulfillment_request.product = product_name
    remove_fulfillment_request.type_ = 'pickup-in-store'
    remove_fulfillment_request.place_ids = [store_id]
    remove_fulfillment_request.allow_missing = True
    print('---remove fulfillment request---')
    print(remove_fulfillment_request)
    return remove_fulfillment_request

async def remove_places(product_name: str):
    print('------remove fulfillment places-----')
    remove_fulfillment_request = get_remove_fulfillment_request(product_name, 'store0')
    operation = ProductServiceClient().remove_fulfillment_places(remove_fulfillment_request)
    try:
        while not operation.done():
            print('---please wait till operation is done---')
            time.sleep(30)
        print('---remove fulfillment places operation is done---')
    except GoogleAPICallError:
        pass
    get_product(product_name)
    delete_product(product_name)
product = create_product(product_id)
asyncio.run(remove_places(product.name))