import re
import subprocess
from google.api_core.retry import Retry
from setup_product.setup_cleanup import create_bq_dataset, create_bq_table, delete_bq_table, upload_data_to_bq_table

@Retry()
def test_import_products_bq(table_id_prefix):
    if False:
        for i in range(10):
            print('nop')
    dataset = 'products'
    valid_products_table = f'{table_id_prefix}products'
    product_schema = '../resources/product_schema.json'
    valid_products_source_file = '../resources/products.json'
    create_bq_dataset(dataset)
    create_bq_table(dataset, valid_products_table, product_schema)
    upload_data_to_bq_table(dataset, valid_products_table, valid_products_source_file, product_schema)
    output = str(subprocess.check_output(f'python import_products_big_query_table.py {dataset} {valid_products_table}', shell=True))
    delete_bq_table(dataset, valid_products_table)
    assert re.match('.*import products from big query table request.*', output)
    assert re.match('.*the operation was started.*', output)
    assert re.match('.*projects/.*/locations/global/catalogs/default_catalog/branches/0/operations/import-products.*', output)
    assert re.match('.*number of successfully imported products.*?316.*', output)
    assert re.match('.*number of failures during the importing.*?0.*', output)