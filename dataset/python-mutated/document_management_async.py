from azure.cosmos.aio import CosmosClient
import azure.cosmos.exceptions as exceptions
from azure.cosmos.http_constants import StatusCodes
from azure.cosmos.partition_key import PartitionKey
import datetime
import asyncio
import config
HOST = config.settings['host']
MASTER_KEY = config.settings['master_key']
DATABASE_ID = config.settings['database_id']
CONTAINER_ID = config.settings['container_id']
CONTAINER_MH_ID = config.settings['container_mh_id']

async def create_items(container):
    print('Creating Items')
    print('\n1.1 Create Item\n')
    sales_order = get_sales_order('SalesOrder1')
    await container.create_item(body=sales_order)
    sales_order2 = get_sales_order_v2('SalesOrder2')
    await container.create_item(body=sales_order2)

async def read_item(container, doc_id):
    print('\n1.2 Reading Item by Id\n')
    response = await container.read_item(item=doc_id, partition_key=doc_id)
    print('Item read by Id {0}'.format(doc_id))
    print('Account Number: {0}'.format(response.get('account_number')))
    print('Subtotal: {0}'.format(response.get('subtotal')))

async def read_items(container):
    print('\n1.3 - Reading all items in a container\n')
    read_all_items_response = container.read_all_items(max_item_count=10)
    item_list = [item async for item in read_all_items_response]
    print('Found {0} items'.format(item_list.__len__()))
    for doc in item_list:
        print('Item Id: {0}'.format(doc.get('id')))
    async for item in read_all_items_response:
        print(item.get('id'))

async def query_items(container, doc_id):
    print('\n1.4 Querying for an  Item by Id\n')
    query_items_response = container.query_items(query='SELECT * FROM r WHERE r.id=@id', parameters=[{'name': '@id', 'value': doc_id}])
    items = [item async for item in query_items_response]
    print('Item queried by Id {0}'.format(items[0].get('id')))

async def query_items_with_continuation_token(container):
    print('\n1.5 Querying for Items using Pagination and Continuation Token\n')
    query_iterable = container.query_items(query='SELECT * FROM r', max_item_count=1)
    item_pages = query_iterable.by_page()
    first_page = await anext(item_pages)
    continuation_token = item_pages.continuation_token
    items_from_continuation = query_iterable.by_page(continuation_token)
    second_page_items_with_continuation = [item async for item in await anext(items_from_continuation)]
    print('The single items in the second page are {}'.format(second_page_items_with_continuation[0].get('id')))

async def replace_item(container, doc_id):
    print('\n1.6 Replace an Item\n')
    read_item = await container.read_item(item=doc_id, partition_key=doc_id)
    read_item['subtotal'] = read_item['subtotal'] + 1
    response = await container.replace_item(item=read_item, body=read_item)
    print("Replaced Item's Id is {0}, new subtotal={1}".format(response['id'], response['subtotal']))

async def upsert_item(container, doc_id):
    print('\n1.7 Upserting an item\n')
    read_item = await container.read_item(item=doc_id, partition_key=doc_id)
    read_item['subtotal'] = read_item['subtotal'] + 1
    response = await container.upsert_item(body=read_item)
    print("Upserted Item's Id is {0}, new subtotal={1}".format(response['id'], response['subtotal']))

async def conditional_patch_item(container, doc_id):
    print('\n1.8 Patching Item by Id based on filter\n')
    operations = [{'op': 'add', 'path': '/favorite_color', 'value': 'red'}, {'op': 'remove', 'path': '/ttl'}, {'op': 'replace', 'path': '/tax_amount', 'value': 14}, {'op': 'set', 'path': '/items/0/discount', 'value': 20.0512}, {'op': 'incr', 'path': '/total_due', 'value': 5}, {'op': 'move', 'from': '/freight', 'path': '/service_addition'}]
    filter_predicate = 'from c where c.tax_amount = 10'
    print('Filter predicate match failure will result in BadRequestException.')
    try:
        await container.patch_item(item=doc_id, partition_key=doc_id, patch_operations=operations, filter_predicate=filter_predicate)
    except exceptions.CosmosHttpResponseError as e:
        assert e.status_code == StatusCodes.PRECONDITION_FAILED
        print('Failed as expected.')

async def patch_item(container, doc_id):
    print('\n1.9 Patching Item by Id\n')
    operations = [{'op': 'add', 'path': '/favorite_color', 'value': 'red'}, {'op': 'remove', 'path': '/ttl'}, {'op': 'replace', 'path': '/tax_amount', 'value': 14}, {'op': 'set', 'path': '/items/0/discount', 'value': 20.0512}, {'op': 'incr', 'path': '/total_due', 'value': 5}, {'op': 'move', 'from': '/freight', 'path': '/service_addition'}]
    response = await container.patch_item(item=doc_id, partition_key=doc_id, patch_operations=operations)
    print("Patched Item's Id is {0}, new path favorite color={1}, removed path ttl={2}, replaced path tax_amount={3}, set path for item at index 0 of discount={4}, increase in path total_due, new total_due={5}, move from path freight={6} to path service_addition={7}".format(response['id'], response['favorite_color'], response.get('ttl'), response['tax_amount'], response['items'][0].get('discount'), response['total_due'], response.get('freight'), response['service_addition']))

async def execute_item_batch(database):
    print('\n1.10 Executing Batch Item operations\n')
    container = await database.create_container_if_not_exists(id='batch_container', partition_key=PartitionKey(path='/account_number'))
    await container.create_item(get_sales_order('read_item'))
    await container.create_item(get_sales_order('delete_item'))
    await container.create_item(get_sales_order('replace_item'))
    create_item_operation = ('create', (get_sales_order('create_item'),))
    upsert_item_operation = ('upsert', (get_sales_order('upsert_item'),))
    read_item_operation = ('read', ('read_item',))
    delete_item_operation = ('delete', ('delete_item',))
    replace_item_operation = ('replace', ('replace_item', {'id': 'replace_item', 'message': 'item was replaced'}))
    replace_item_if_match_operation = ('replace', ('replace_item', {'id': 'replace_item', 'message': 'item was replaced'}), {'if_match_etag': container.client_connection.last_response_headers.get('etag')})
    replace_item_if_none_match_operation = ('replace', ('replace_item', {'id': 'replace_item', 'message': 'item was replaced'}), {'if_none_match_etag': container.client_connection.last_response_headers.get('etag')})
    batch_operations = [create_item_operation, upsert_item_operation, read_item_operation, delete_item_operation, replace_item_operation, replace_item_if_match_operation, replace_item_if_none_match_operation]
    batch_results = await container.execute_item_batch(batch_operations=batch_operations, partition_key='Account1')
    print('\nResults for the batch operations: {}\n'.format(batch_results))
    with open('file_name.txt', 'r') as data_file:
        container.execute_item_batch([('upsert', (t,)) for t in data_file.readlines()])
    batch_operations = [create_item_operation, create_item_operation]
    try:
        await container.execute_item_batch(batch_operations, partition_key='Account1')
    except exceptions.CosmosBatchOperationError as e:
        error_operation_index = e.error_index
        error_operation_response = e.operation_responses[error_operation_index]
        error_operation = batch_operations[error_operation_index]
        print('\nError operation: {}, error operation response: {}\n'.format(error_operation, error_operation_response))

async def delete_item(container, doc_id):
    print('\n1.11 Deleting Item by Id\n')
    await container.delete_item(item=doc_id, partition_key=doc_id)
    print("Deleted item's Id is {0}".format(doc_id))

async def delete_all_items_by_partition_key(db, partitionkey):
    print('\n1.12 Deleting all Items by Partition Key\n')
    container = await db.create_container_if_not_exists(id='Partition Key Delete Container', partition_key=PartitionKey(path='/company'))
    sales_order_company_A1 = get_sales_order('SalesOrderCompanyA1')
    sales_order_company_A1['company'] = partitionkey
    await container.upsert_item(sales_order_company_A1)
    print('\nUpserted Item is {} with Partition Key: {}'.format(sales_order_company_A1['id'], partitionkey))
    sales_order_company_A2 = get_sales_order('SalesOrderCompanyA2')
    sales_order_company_A2['company'] = partitionkey
    await container.upsert_item(sales_order_company_A2)
    print('\nUpserted Item is {} with Partition Key: {}'.format(sales_order_company_A2['id'], partitionkey))
    sales_order_company_B1 = get_sales_order('SalesOrderCompanyB1')
    sales_order_company_B1['company'] = 'companyB'
    await container.upsert_item(sales_order_company_B1)
    print('\nUpserted Item is {} with Partition Key: {}'.format(sales_order_company_B1['id'], 'companyB'))
    item_list = [item async for item in container.read_all_items()]
    print('Found {0} items'.format(item_list.__len__()))
    for doc in item_list:
        print('Item Id: {0}; Partition Key: {1}'.format(doc.get('id'), doc.get('company')))
    print('\nDelete all items for Partition Key: {}\n'.format(partitionkey))
    await container.delete_all_items_by_partition_key(partitionkey)
    item_list = [item async for item in container.read_all_items()]
    print('Found {0} items'.format(item_list.__len__()))
    for doc in item_list:
        print('Item Id: {0}; Partition Key: {1}'.format(doc.get('id'), doc.get('company')))

async def create_mh_items(container):
    print('Creating Items')
    print('\n1.13 Create Item with Multi Hash Partition Key\n')
    sales_order = get_sales_order('SalesOrder1')
    await container.create_item(body=sales_order)
    sales_order2 = get_sales_order_v2('SalesOrder2')
    await container.create_item(body=sales_order2)

async def read_mh_item(container, doc_id, pk):
    print('\n1.14 Reading Item by Multi Hash Partition Key\n')
    response = await container.read_item(item=doc_id, partition_key=pk)
    print('Item read by Partition Key {0}'.format(pk))
    print('Account Number: {0}'.format(response.get('account_number')))
    print('Purchase Order Number: {0}'.format(response.get('purchase_order_number')))

async def query_mh_items(container, pk):
    print('\n1.15 Querying for an  Item by Multi Hash Partition Key\n')
    query_items_response = container.query_items(query='SELECT * FROM r WHERE r.account_number=@account_number and r.purchase_order_number=@purchase_order_number', parameters=[{'name': '@account_number', 'value': pk[0]}, {'name': '@purchase_order_number', 'value': pk[1]}])
    items = [item async for item in query_items_response]
    print('Account Number: {0}'.format(items[0].get('account_number')))
    print('Purchase Order Number: {0}'.format(items[0].get('purchase_order_number')))

async def replace_mh_item(container, doc_id, pk):
    print('\n1.16 Replace an Item with Multi Hash Partition Key\n')
    read_item = await container.read_item(item=doc_id, partition_key=pk)
    read_item['subtotal'] = read_item['subtotal'] + 1
    response = await container.replace_item(item=read_item, body=read_item)
    print("Replaced Item's Account Number is {0}, Purchase Order Number is {1}, new subtotal={2}".format(response['account_number'], response['purchase_order_number'], response['subtotal']))

async def upsert_mh_item(container, doc_id, pk):
    print('\n1.17 Upserting an item with Multi Hash Partition Key\n')
    read_item = await container.read_item(item=doc_id, partition_key=pk)
    read_item['subtotal'] = read_item['subtotal'] + 1
    response = await container.upsert_item(body=read_item)
    print("Replaced Item's Account Number is {0}, Purchase Order Number is {1}, new subtotal={2}".format(response['account_number'], response['purchase_order_number'], response['subtotal']))

async def patch_mh_item(container, doc_id, pk):
    print('\n1.18 Patching Item by Multi Hash Partition Key\n')
    operations = [{'op': 'add', 'path': '/favorite_color', 'value': 'red'}, {'op': 'remove', 'path': '/ttl'}, {'op': 'replace', 'path': '/tax_amount', 'value': 14}, {'op': 'set', 'path': '/items/0/discount', 'value': 20.0512}, {'op': 'incr', 'path': '/total_due', 'value': 5}, {'op': 'move', 'from': '/freight', 'path': '/service_addition'}]
    response = await container.patch_item(item=doc_id, partition_key=pk, patch_operations=operations)
    print("Patched Item's Id is {0}, new path favorite color={1}, removed path ttl={2}, replaced path tax_amount={3}, set path for item at index 0 of discount={4}, increase in path total_due, new total_due={5}, move from path freight={6} to path service_addition={7}".format(response['id'], response['favorite_color'], response.get('ttl'), response['tax_amount'], response['items'][0].get('discount'), response['total_due'], response.get('freight'), response['service_addition']))

async def delete_mh_item(container, doc_id, pk):
    print('\n1.19 Deleting Item by Multi Hash Partition Key\n')
    response = await container.delete_item(item=doc_id, partition_key=pk)
    print("Deleted item's Account Number is {0} Purchase Order Number is {1}".format(pk[0], pk[1]))

async def delete_all_items_by_partition_key_mh(db, partitionkey):
    print('\n1.20 Deleting all Items by Partition Key Multi Hash\n')
    container = await db.create_container_if_not_exists(id='Partition Key Delete Container Multi Hash', partition_key=PartitionKey(path=['/id', '/company'], kind='MultiHash'))
    sales_order_company_A1 = get_sales_order(partitionkey[0])
    sales_order_company_A1['company'] = partitionkey[1]
    await container.upsert_item(sales_order_company_A1)
    print('\nUpserted Item is {} with Partition Key: {}'.format(sales_order_company_A1['id'], partitionkey))
    sales_order_company_A2 = get_sales_order(partitionkey[0])
    sales_order_company_A2['company'] = partitionkey[1]
    await container.upsert_item(sales_order_company_A2)
    print('\nUpserted Item is {} with Partition Key: {}'.format(sales_order_company_A2['id'], partitionkey))
    sales_order_company_B1 = get_sales_order('SalesOrderCompanyB1')
    sales_order_company_B1['company'] = 'companyB'
    await container.upsert_item(sales_order_company_B1)
    print('\nUpserted Item is {} with Partition Key: {}'.format(sales_order_company_B1['id'], 'companyB'))
    item_list = [item async for item in container.read_all_items()]
    print('Found {0} items'.format(item_list.__len__()))
    for doc in item_list:
        print('Item Id: {0}; Partition Key: {1}'.format(doc.get('id'), doc.get('company')))
    print('\nDelete all items for Partition Key: {}\n'.format(partitionkey))
    await container.delete_all_items_by_partition_key(partitionkey)
    item_list = [item async for item in container.read_all_items()]
    print('Found {0} items'.format(item_list.__len__()))
    for doc in item_list:
        print('Item Id: {0}; Partition Key: {1}'.format(doc.get('id'), doc.get('company')))

async def query_items_with_continuation_token_size_limit(container, doc_id):
    print('\n1.21 Query Items With Continuation Token Size Limit.\n')
    size_limit_in_kb = 8
    sales_order = get_sales_order(doc_id)
    await container.create_item(body=sales_order)
    items = container.query_items(query='SELECT * FROM r', partition_key=doc_id, continuation_token_limit=size_limit_in_kb)
    print('Continuation Token size has been limited to {}KB.'.format(size_limit_in_kb))

def get_sales_order(item_id):
    if False:
        while True:
            i = 10
    order1 = {'id': item_id, 'account_number': 'Account1', 'purchase_order_number': 'PO18009186470', 'order_date': datetime.date(2005, 1, 10).strftime('%c'), 'subtotal': 419.4589, 'tax_amount': 12.5838, 'freight': 472.3108, 'total_due': 985.018, 'items': [{'order_qty': 1, 'product_id': 100, 'unit_price': 418.4589, 'line_price': 418.4589}], 'ttl': 60 * 60 * 24 * 30}
    return order1

def get_sales_order_v2(item_id):
    if False:
        i = 10
        return i + 15
    order2 = {'id': item_id, 'account_number': 'Account2', 'purchase_order_number': 'PO15428132599', 'order_date': datetime.date(2005, 7, 11).strftime('%c'), 'due_date': datetime.date(2005, 7, 21).strftime('%c'), 'shipped_date': datetime.date(2005, 7, 15).strftime('%c'), 'subtotal': 6107.082, 'tax_amount': 586.1203, 'freight': 183.1626, 'discount_amt': 1982.872, 'total_due': 4893.3929, 'items': [{'order_qty': 3, 'product_code': 'A-123', 'product_name': 'Product 1', 'currency_symbol': '$', 'currency_code': 'USD', 'unit_price': 17.1, 'line_price': 5.7}], 'ttl': 60 * 60 * 24 * 30}
    return order2

async def run_sample():
    async with CosmosClient(HOST, {'masterKey': MASTER_KEY}) as client:
        try:
            db = await client.create_database_if_not_exists(id=DATABASE_ID)
            container = await db.create_container_if_not_exists(id=CONTAINER_ID, partition_key=PartitionKey(path='/id', kind='Hash'))
            print("Container with id '{0}' created".format(CONTAINER_ID))
            await create_items(container)
            await read_item(container, 'SalesOrder1')
            await read_items(container)
            await query_items(container, 'SalesOrder1')
            await query_items_with_continuation_token(container)
            await replace_item(container, 'SalesOrder1')
            await upsert_item(container, 'SalesOrder1')
            await conditional_patch_item(container, 'SalesOrder1')
            await patch_item(container, 'SalesOrder1')
            await delete_item(container, 'SalesOrder1')
            await delete_all_items_by_partition_key(db, 'CompanyA')
            await query_items_with_continuation_token_size_limit(container, 'SalesOrder1')
            container_multi_hash = await db.create_container_if_not_exists(id=CONTAINER_MH_ID, partition_key=PartitionKey(path=['/account_number', '/purchase_order_number'], kind='MultiHash'))
            await create_mh_items(container_multi_hash)
            await read_mh_item(container_multi_hash, 'SalesOrder1', ['Account1', 'PO18009186470'])
            await query_mh_items(container_multi_hash, ['Account1', 'PO18009186470'])
            await replace_mh_item(container_multi_hash, 'SalesOrder1', ['Account1', 'PO18009186470'])
            await upsert_mh_item(container_multi_hash, 'SalesOrder1', ['Account1', 'PO18009186470'])
            await patch_mh_item(container_multi_hash, 'SalesOrder1', ['Account1', 'PO18009186470'])
            await delete_mh_item(container_multi_hash, 'SalesOrder1', ['Account1', 'PO18009186470'])
            await delete_all_items_by_partition_key_mh(db, ['SalesOrderCompany', 'CompanyA'])
            try:
                await client.delete_database(db)
            except exceptions.CosmosResourceNotFoundError:
                pass
        except exceptions.CosmosHttpResponseError as e:
            print('\nrun_sample has caught an error. {0}'.format(e.message))
        finally:
            print('\nrun_sample done')
if __name__ == '__main__':
    asyncio.run(run_sample())