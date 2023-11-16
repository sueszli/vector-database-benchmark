import os
import uuid
from google.cloud import bigquery
import pytest
client = bigquery.Client()
dataset_id = f'bqml_tutorial_{str(uuid.uuid4().hex)}'
full_dataset_id = f'{client.project}.{dataset_id}'

@pytest.fixture
def delete_dataset():
    if False:
        return 10
    yield
    client.delete_dataset(full_dataset_id, delete_contents=True)

def test_ncaa_tutorial(delete_dataset):
    if False:
        i = 10
        return i + 15
    dataset = bigquery.Dataset(full_dataset_id)
    dataset.location = 'US'
    client.create_dataset(dataset)
    query_files = ['feature_input_query.sql', 'training_data_query.sql']
    resources_directory = os.path.join(os.path.dirname(__file__), 'resources')
    for fname in query_files:
        query_filepath = os.path.join(resources_directory, fname)
        sql = open(query_filepath, encoding='utf-8').read().format(dataset_id)
        client.query(sql).result()
    sql = "\n        CREATE OR REPLACE MODEL `{0}.ncaa_model`\n        OPTIONS (\n            model_type='linear_reg',\n            max_iteration=50 ) AS\n        SELECT\n            * EXCEPT (\n                game_id, season, scheduled_date,\n                total_three_points_made,\n                total_three_points_att),\n            total_three_points_att as label\n        FROM\n            `{0}.wide_games`\n        WHERE\n            # remove the game to predict\n            game_id != 'f1063e80-23c7-486b-9a5e-faa52beb2d83'\n    ".format(dataset_id)
    df = client.query(sql).to_dataframe()
    print(df)
    sql = '\n        SELECT\n            *\n        FROM\n            ML.TRAINING_INFO(MODEL `{}.ncaa_model`)\n    '.format(dataset_id)
    df = client.query(sql).to_dataframe()
    print(df)
    sql = '\n        WITH eval_table AS (\n            SELECT\n                *,\n                total_three_points_att AS label\n            FROM\n                `{0}.wide_games` )\n        SELECT\n            *\n        FROM\n            ML.EVALUATE(MODEL `{0}.ncaa_model`,\n                TABLE eval_table)\n    '.format(dataset_id)
    df = client.query(sql).to_dataframe()
    print(df)
    sql = "\n        WITH game_to_predict AS (\n            SELECT\n                *\n            FROM\n                `{0}.wide_games`\n            WHERE\n                game_id='f1063e80-23c7-486b-9a5e-faa52beb2d83' )\n        SELECT\n            truth.game_id AS game_id,\n            total_three_points_att,\n            predicted_total_three_points_att\n        FROM (\n            SELECT\n                game_id,\n                predicted_label AS predicted_total_three_points_att\n            FROM\n                ML.PREDICT(MODEL `{0}.ncaa_model`,\n                table game_to_predict) ) AS predict\n        JOIN (\n            SELECT\n                game_id,\n                total_three_points_att AS total_three_points_att\n            FROM\n                game_to_predict) AS truth\n        ON\n            predict.game_id = truth.game_id\n    ".format(dataset_id)
    df = client.query(sql).to_dataframe()
    print(df)