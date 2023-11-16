import requests
import csv
import argparse
from pyspark.sql.types import StructType, StructField, FloatType, TimestampType
from bigdl.orca.common import init_orca_context, stop_orca_context, OrcaContext
from bigdl.chronos.data.experimental import XShardsTSDataset
from bigdl.chronos.forecaster import TCNForecaster
from bigdl.chronos.metric.forecast_metrics import Evaluator
import numpy as np
from sklearn.preprocessing import StandardScaler
from urllib.parse import urlparse
from os.path import exists
from bigdl.dllib.utils import log4Error

def is_local_and_existing_uri(uri):
    if False:
        i = 10
        return i + 15
    parsed_uri = urlparse(uri)
    log4Error.invalidInputError(not parsed_uri.scheme or parsed_uri.scheme == 'file', 'Not Local File!')
    log4Error.invalidInputError(not parsed_uri.netloc or parsed_uri.netloc.lower() == 'localhost', 'Not Local File!')
    log4Error.invalidInputError(exists(parsed_uri.path), 'File Not Exist!')

def generate_spark_df(dataset_path):
    if False:
        while True:
            i = 10
    spark = OrcaContext.get_spark_session()
    schema = StructType([StructField('timestamp', TimestampType()), StructField('value', FloatType())])
    df = spark.read.format('csv').schema(schema).option('header', 'true').load(dataset_path)
    (tsdata_train, _, tsdata_test) = XShardsTSDataset.from_sparkdf(df, dt_col='timestamp', target_col=['value'], with_split=True, val_ratio=0, test_ratio=0.1)
    scaler = {'0': StandardScaler()}
    for tsdata in [tsdata_train, tsdata_test]:
        tsdata.scale(scaler, fit=tsdata is tsdata_train).roll(lookback=100, horizon=1)
    return (tsdata_train, tsdata_test)

def get_csv(args):
    if False:
        i = 10
        return i + 15
    dataset_path = args.datadir
    if args.datadir is None:
        with requests.get(args.url) as r:
            data = (line.decode('utf-8') for line in r.iter_lines())
            data = csv.reader(data, delimiter=',')
            f = open('nyc_taxi.csv', 'w', encoding='utf-8', newline='')
            csv_writer = csv.writer(f)
            for row in data:
                csv_writer.writerow(row)
            f.close()
            dataset_path = 'nyc_taxi.csv'
    is_local_and_existing_uri(dataset_path)
    return dataset_path
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cores', type=int, default=4, help='The number of cpu cores you want to use on each node.You can change it depending on your own cluster setting.')
    parser.add_argument('--memory', type=str, default='32g', help='The memory you want to use on each node.You can change it depending on your own cluster setting.')
    parser.add_argument('--cluster_mode', type=str, default='local', help='The mode for the Spark cluster.')
    parser.add_argument('--num_nodes', type=int, default=1, help='The number of nodes to be used in the clusterYou can change it depending on your own cluster setting.')
    parser.add_argument('--epochs', type=int, default=2, help='Max number of epochs to train in each trial.')
    parser.add_argument('--workers_per_node', type=int, default=1, help='the number of worker you want to use.The value defaults to 1. The param is only effectivewhen distributed is set to True.')
    parser.add_argument('--datadir', type=str, help='Use local csv file by default.')
    parser.add_argument('--url', type=str, default='https://raw.githubusercontent.com/numenta/NAB/v1.0/data/realKnownCause/nyc_taxi.csv', help='Download link of dataset.')
    args = parser.parse_args()
    num_nodes = 1 if args.cluster_mode == 'local' else args.num_nodes
    init_orca_context(cluster_mode=args.cluster_mode, cores=args.cores, memory=args.memory, num_nodes=num_nodes)
    dataset_path = get_csv(args)
    (tsdata_train, tsdata_test) = generate_spark_df(dataset_path)
    data_train = tsdata_train.to_xshards()
    data_test = tsdata_test.to_xshards()
    forecaster = TCNForecaster(past_seq_len=100, future_seq_len=1, input_feature_num=1, output_feature_num=1, metrics=['mse', 'mae'], distributed=True, workers_per_node=args.workers_per_node, seed=1)
    forecaster.fit(data_train, epochs=args.epochs)
    yhat = forecaster.predict(data_test)
    yhat_unscaled = tsdata_test.unscale_xshards(yhat, key='prediction').collect()
    y_unscaled = tsdata_test.unscale_xshards(data_test, key='y').collect()
    print('mae for test dataset:', Evaluator.evaluate(['mae'], np.concatenate([y_unscaled[i] for i in range(len(y_unscaled))], axis=0), np.concatenate([yhat_unscaled[i] for i in range(len(yhat_unscaled))], axis=0)))
    stop_orca_context()