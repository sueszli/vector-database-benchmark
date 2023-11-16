import time
import argparse
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from bigdl.chronos.forecaster.seq2seq_forecaster import Seq2SeqForecaster
from bigdl.chronos.data import get_public_dataset

def get_tsdata():
    if False:
        return 10
    name = 'network_traffic'
    (tsdata_train, _, tsdata_test) = get_public_dataset(name, val_ratio=0)
    minmax = MinMaxScaler()
    for tsdata in [tsdata_train, tsdata_test]:
        tsdata.gen_dt_feature(one_hot_features=['HOUR', 'WEEK']).impute('last').scale(minmax, fit=tsdata is tsdata_train)
    return (tsdata_train, tsdata_test)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2, help='Max number of epochs to train in each trial.')
    args = parser.parse_args()
    (tsdata_train, tsdata_test) = get_tsdata()
    forecaster = Seq2SeqForecaster(past_seq_len=100, future_seq_len=10, input_feature_num=33, output_feature_num=2, metrics=['mse', 'smape'], seed=0)
    (x_train, y_train) = tsdata_train.roll(lookback=100, horizon=10).to_numpy()
    (x_test, y_test) = tsdata_test.roll(lookback=100, horizon=10).to_numpy()
    forecaster.fit((x_train, y_train), epochs=args.epochs)
    (mse, smape) = forecaster.evaluate((x_test, y_test))
    print(f'evaluate mse is: {np.mean(mse):.4f}')
    print(f'evaluate smape is: {np.mean(smape):.4f}')
    (mse, smape) = forecaster.evaluate_with_onnx((x_test, y_test))
    print(f'evaluate_onnx mse is: {np.mean(mse):.4f}')
    print(f'evaluate_onnx smape is: {np.mean(smape):.4f}')
    start_time = time.time()
    forecaster.predict(x_test)
    print(f'inference time is: {time.time() - start_time:.3f}s')
    start_time = time.time()
    forecaster.predict_with_onnx(x_test)
    print(f'inference(onnx) time is: {time.time() - start_time:.3f}s')