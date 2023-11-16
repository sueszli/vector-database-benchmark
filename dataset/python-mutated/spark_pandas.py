import sys
from optparse import OptionParser
import bigdl.orca.data.pandas
from bigdl.orca import init_orca_context, stop_orca_context

def process_feature(df, awake_begin=6, awake_end=23):
    if False:
        return 10
    import pandas as pd
    df['datetime'] = pd.to_datetime(df['timestamp'])
    df['hours'] = df['datetime'].dt.hour
    df['awake'] = ((df['hours'] >= awake_begin) & (df['hours'] <= awake_end) | (df['hours'] == 0)).astype(int)
    return df
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-f', type=str, dest='file_path', help='The file path to be read')
    parser.add_option('--deploy-mode', type=str, dest='deployMode', default='local', help='deploy mode, local, spark-submit, yarn-client or yarn-cluster')
    (options, args) = parser.parse_args(sys.argv)
    sc = init_orca_context(cluster_mode=options.deployMode)
    file_path = options.file_path
    data_shard = bigdl.orca.data.pandas.read_csv(file_path)
    data = data_shard.collect()
    data_shard = data_shard.repartition(2)
    trans_data_shard = data_shard.transform_shard(process_feature)
    data2 = trans_data_shard.collect()
    from bigdl.orca.data.transformer import *
    encode = StringIndexer(inputCol='value')
    encoded_data_shard = encode.fit_transform(trans_data_shard)
    trans_data_shard = trans_data_shard.deduplicates()
    scale = MinMaxScaler(inputCol=['hours', 'awake'], outputCol='x_scaled')
    data_shard = scale.fit_transform(trans_data_shard)
    scale = MinMaxScaler(inputCol=['hours'], outputCol='x_scaled')
    data_shard = scale.fit_transform(trans_data_shard)
    stop_orca_context()