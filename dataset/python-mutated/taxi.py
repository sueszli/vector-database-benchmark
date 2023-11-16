"""Utility and schema methods for the chicago_taxi sample."""
from tensorflow_transform import coders as tft_coders
from tensorflow_transform.tf_metadata import schema_utils
from google.protobuf import text_format
from tensorflow.python.lib.io import file_io
from tensorflow_metadata.proto.v0 import schema_pb2
MAX_CATEGORICAL_FEATURE_VALUES = [24, 31, 12]
CATEGORICAL_FEATURE_KEYS = ['trip_start_hour', 'trip_start_day', 'trip_start_month', 'pickup_census_tract', 'dropoff_census_tract', 'pickup_community_area', 'dropoff_community_area']
DENSE_FLOAT_FEATURE_KEYS = ['trip_miles', 'fare', 'trip_seconds']
FEATURE_BUCKET_COUNT = 10
BUCKET_FEATURE_KEYS = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
VOCAB_SIZE = 1000
OOV_SIZE = 10
VOCAB_FEATURE_KEYS = ['payment_type', 'company']
LABEL_KEY = 'tips'
FARE_KEY = 'fare'
CSV_COLUMN_NAMES = ['pickup_community_area', 'fare', 'trip_start_month', 'trip_start_hour', 'trip_start_day', 'trip_start_timestamp', 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'trip_miles', 'pickup_census_tract', 'dropoff_census_tract', 'payment_type', 'company', 'trip_seconds', 'dropoff_community_area', 'tips']

def transformed_name(key):
    if False:
        for i in range(10):
            print('nop')
    return key + '_xf'

def transformed_names(keys):
    if False:
        while True:
            i = 10
    return [transformed_name(key) for key in keys]

def get_raw_feature_spec(schema):
    if False:
        i = 10
        return i + 15
    return schema_utils.schema_as_feature_spec(schema).feature_spec

def make_proto_coder(schema):
    if False:
        while True:
            i = 10
    raw_feature_spec = get_raw_feature_spec(schema)
    raw_schema = schema_utils.schema_from_feature_spec(raw_feature_spec)
    return tft_coders.ExampleProtoCoder(raw_schema)

def make_csv_coder(schema):
    if False:
        print('Hello World!')
    'Return a coder for tf.transform to read csv files.'
    raw_feature_spec = get_raw_feature_spec(schema)
    parsing_schema = schema_utils.schema_from_feature_spec(raw_feature_spec)
    return tft_coders.CsvCoder(CSV_COLUMN_NAMES, parsing_schema)

def clean_raw_data_dict(input_dict, raw_feature_spec):
    if False:
        i = 10
        return i + 15
    'Clean raw data dict.'
    output_dict = {}
    for key in raw_feature_spec:
        if key not in input_dict or not input_dict[key]:
            output_dict[key] = []
        else:
            output_dict[key] = [input_dict[key]]
    return output_dict

def make_sql(table_name, max_rows=None, for_eval=False):
    if False:
        for i in range(10):
            print('nop')
    'Creates the sql command for pulling data from BigQuery.\n\n  Args:\n    table_name: BigQuery table name\n    max_rows: if set, limits the number of rows pulled from BigQuery\n    for_eval: True if this is for evaluation, false otherwise\n\n  Returns:\n    sql command as string\n  '
    if for_eval:
        where_clause = 'WHERE MOD(FARM_FINGERPRINT(unique_key), 3) = 0 AND pickup_latitude is not null AND pickup_longitude is not null AND dropoff_latitude is not null AND dropoff_longitude is not null'
    else:
        where_clause = 'WHERE MOD(FARM_FINGERPRINT(unique_key), 3) > 0 AND pickup_latitude is not null AND pickup_longitude is not null AND dropoff_latitude is not null AND dropoff_longitude is not null'
    limit_clause = ''
    if max_rows:
        limit_clause = 'LIMIT {max_rows}'.format(max_rows=max_rows)
    return '\n  SELECT\n      CAST(pickup_community_area AS string) AS pickup_community_area,\n      CAST(dropoff_community_area AS string) AS dropoff_community_area,\n      CAST(pickup_census_tract AS string) AS pickup_census_tract,\n      CAST(dropoff_census_tract AS string) AS dropoff_census_tract,\n      fare,\n      EXTRACT(MONTH FROM trip_start_timestamp) AS trip_start_month,\n      EXTRACT(HOUR FROM trip_start_timestamp) AS trip_start_hour,\n      EXTRACT(DAYOFWEEK FROM trip_start_timestamp) AS trip_start_day,\n      UNIX_SECONDS(trip_start_timestamp) AS trip_start_timestamp,\n      pickup_latitude,\n      pickup_longitude,\n      dropoff_latitude,\n      dropoff_longitude,\n      trip_miles,\n      payment_type,\n      company,\n      trip_seconds,\n      tips\n  FROM `{table_name}`\n  {where_clause}\n  {limit_clause}\n'.format(table_name=table_name, where_clause=where_clause, limit_clause=limit_clause)

def read_schema(path):
    if False:
        return 10
    'Reads a schema from the provided location.\n\n  Args:\n    path: The location of the file holding a serialized Schema proto.\n\n  Returns:\n    An instance of Schema or None if the input argument is None\n  '
    result = schema_pb2.Schema()
    contents = file_io.read_file_to_string(path)
    text_format.Parse(contents, result)
    return result