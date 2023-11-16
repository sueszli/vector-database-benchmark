import uuid
from google.cloud import bigquery
import pytest
client = bigquery.Client()
dataset_id = f'bqml_tutorial_{str(uuid.uuid4().hex)}'
full_dataset_id = f'{client.project}.{dataset_id}'

@pytest.fixture
def delete_dataset():
    if False:
        print('Hello World!')
    yield
    client.delete_dataset(full_dataset_id, delete_contents=True)

def test_data_scientist_tutorial(delete_dataset):
    if False:
        i = 10
        return i + 15
    dataset = bigquery.Dataset(full_dataset_id)
    dataset.location = 'US'
    client.create_dataset(dataset)
    sql = '\n        CREATE OR REPLACE MODEL `{}.sample_model`\n        OPTIONS(model_type=\'logistic_reg\') AS\n        SELECT\n            IF(totals.transactions IS NULL, 0, 1) AS label,\n            IFNULL(device.operatingSystem, "") AS os,\n            device.isMobile AS is_mobile,\n            IFNULL(geoNetwork.country, "") AS country,\n            IFNULL(totals.pageviews, 0) AS pageviews\n        FROM\n            `bigquery-public-data.google_analytics_sample.ga_sessions_*`\n        WHERE\n            _TABLE_SUFFIX BETWEEN \'20160801\' AND \'20170630\'\n    '.format(dataset_id)
    df = client.query(sql).to_dataframe()
    print(df)
    sql = '\n        SELECT\n        *\n        FROM\n        ML.TRAINING_INFO(MODEL `{}.sample_model`)\n    '.format(dataset_id)
    df = client.query(sql).to_dataframe()
    print(df)
    sql = '\n        SELECT\n            *\n        FROM ML.EVALUATE(MODEL `{}.sample_model`, (\n            SELECT\n                IF(totals.transactions IS NULL, 0, 1) AS label,\n                IFNULL(device.operatingSystem, "") AS os,\n                device.isMobile AS is_mobile,\n                IFNULL(geoNetwork.country, "") AS country,\n                IFNULL(totals.pageviews, 0) AS pageviews\n            FROM\n                `bigquery-public-data.google_analytics_sample.ga_sessions_*`\n            WHERE\n                _TABLE_SUFFIX BETWEEN \'20170701\' AND \'20170801\'))\n    '.format(dataset_id)
    df = client.query(sql).to_dataframe()
    print(df)
    sql = '\n        SELECT\n            country,\n            SUM(predicted_label) as total_predicted_purchases\n        FROM ML.PREDICT(MODEL `{}.sample_model`, (\n            SELECT\n                IFNULL(device.operatingSystem, "") AS os,\n                device.isMobile AS is_mobile,\n                IFNULL(totals.pageviews, 0) AS pageviews,\n                IFNULL(geoNetwork.country, "") AS country\n            FROM\n                `bigquery-public-data.google_analytics_sample.ga_sessions_*`\n            WHERE\n                _TABLE_SUFFIX BETWEEN \'20170701\' AND \'20170801\'))\n            GROUP BY country\n            ORDER BY total_predicted_purchases DESC\n            LIMIT 10\n    '.format(dataset_id)
    df = client.query(sql).to_dataframe()
    print(df)
    sql = '\n        SELECT\n            fullVisitorId,\n            SUM(predicted_label) as total_predicted_purchases\n        FROM ML.PREDICT(MODEL `{}.sample_model`, (\n            SELECT\n                IFNULL(device.operatingSystem, "") AS os,\n                device.isMobile AS is_mobile,\n                IFNULL(totals.pageviews, 0) AS pageviews,\n                IFNULL(geoNetwork.country, "") AS country,\n                fullVisitorId\n            FROM\n                `bigquery-public-data.google_analytics_sample.ga_sessions_*`\n            WHERE\n                _TABLE_SUFFIX BETWEEN \'20170701\' AND \'20170801\'))\n            GROUP BY fullVisitorId\n            ORDER BY total_predicted_purchases DESC\n            LIMIT 10\n    '.format(dataset_id)
    df = client.query(sql).to_dataframe()
    print(df)