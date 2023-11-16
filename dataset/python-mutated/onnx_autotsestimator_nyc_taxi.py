import time
import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.chronos.autots.autotsestimator import AutoTSEstimator
from bigdl.chronos.data import get_public_dataset

def get_tsdata():
    if False:
        print('Hello World!')
    name = 'nyc_taxi'
    (tsdata_train, tsdata_val, tsdata_test) = get_public_dataset(name)
    stand = StandardScaler()
    for tsdata in [tsdata_train, tsdata_val, tsdata_test]:
        tsdata.gen_dt_feature(one_hot_features=['HOUR', 'WEEK']).impute('last').scale(stand, fit=tsdata is tsdata_train)
    return (tsdata_train, tsdata_val, tsdata_test)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=2, help='The number of nodes to be used in the cluster. You can change it depending on your own cluster setting.')
    parser.add_argument('--cluster_mode', type=str, default='local', help='The mode for the Spark cluster.')
    parser.add_argument('--cores', type=int, default=4, help='The number of cpu cores you want to use on each node.You can change it depending on your own cluster setting.')
    parser.add_argument('--memory', type=str, default='10g', help='The memory you want to use on each node.You can change it depending on your own cluster setting.')
    parser.add_argument('--epochs', type=int, default=2, help='Max number of epochs to train in each trial.')
    parser.add_argument('--n_sampling', type=int, default=6, help='Number of times to sample from the search_space.')
    args = parser.parse_args()
    num_nodes = 1 if args.cluster_mode == 'local' else args.num_workers
    init_orca_context(cluster_mode=args.cluster_mode, cores=args.cores, memory=args.memory, num_nodes=num_nodes, init_ray_on_spark=True)
    (tsdata_train, tsdata_val, tsdata_test) = get_tsdata()
    autoest = AutoTSEstimator(model='lstm', search_space='normal', past_seq_len=40, future_seq_len=1, cpus_per_trial=2, metric='mse', name='auto_lstm')
    tsppl = autoest.fit(data=tsdata_train, validation_data=tsdata_val, epochs=args.epochs, n_sampling=args.n_sampling)
    (mse, smape) = tsppl.evaluate(tsdata_test, metrics=['mse', 'smape'])
    print(f'evaluate mse is: {np.mean(mse)}')
    print(f'evaluate smape is: {np.mean(smape)}')
    (mse, smape) = tsppl.evaluate_with_onnx(tsdata_test, metrics=['mse', 'smape'])
    print(f'evaluate_onnx mse is: {np.mean(mse)}')
    print(f'evaluate_onnx smape is: {np.mean(smape)}')
    start_time = time.time()
    tsppl.predict(tsdata_test)
    print(f'inference time is: {time.time() - start_time:.3f}s')
    start_time = time.time()
    tsppl.predict_with_onnx(tsdata_test)
    print(f'inference(onnx) time is: {time.time() - start_time:.3f}s')
    stop_orca_context()