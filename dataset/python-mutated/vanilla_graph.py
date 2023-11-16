from pandas import DataFrame
from dagster import Definitions, job, op
from .mylib import create_db_connection, fetch_products

@op
def extract_products() -> DataFrame:
    if False:
        i = 10
        return i + 15
    return fetch_products()

@op
def get_categories(products: DataFrame) -> DataFrame:
    if False:
        i = 10
        return i + 15
    return DataFrame({'category': products['category'].unique()})

@op
def write_products_table(products: DataFrame) -> None:
    if False:
        return 10
    products.to_sql(name='products', con=create_db_connection())

@op
def write_categories_table(categories: DataFrame) -> None:
    if False:
        while True:
            i = 10
    categories.to_sql(name='categories', con=create_db_connection())

@job
def ingest_products_and_categories():
    if False:
        while True:
            i = 10
    products = extract_products()
    product_categories = get_categories(products)
    return (write_products_table(products), write_categories_table(product_categories))
defs = Definitions(jobs=[ingest_products_and_categories])