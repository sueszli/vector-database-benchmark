from unittest import TestCase
import pytest
from bigdl.chronos.utils import LazyImport
torch = LazyImport('torch')
tf = LazyImport('tensorflow')
AutoTSEstimator = LazyImport('bigdl.chronos.autots.autotsestimator.AutoTSEstimator')
TSPipeline = LazyImport('bigdl.chronos.autots.tspipeline.TSPipeline')
hp = LazyImport('bigdl.orca.automl.hp')
import numpy as np
from bigdl.chronos.data import TSDataset
import pandas as pd
from .. import op_torch, op_tf2, op_distributed, op_inference

def get_ts_df():
    if False:
        while True:
            i = 10
    sample_num = np.random.randint(100, 200)
    train_df = pd.DataFrame({'datetime': pd.date_range('1/1/2019', periods=sample_num), 'value 1': np.random.randn(sample_num), 'value 2': np.random.randn(sample_num), 'id': np.array(['00'] * sample_num), 'extra feature 1': np.random.randn(sample_num), 'extra feature 2': np.random.randn(sample_num)})
    return train_df

def get_tsdataset():
    if False:
        for i in range(10):
            print('nop')
    df = get_ts_df()
    return TSDataset.from_pandas(df, dt_col='datetime', target_col=['value 1', 'value 2'], extra_feature_col=['extra feature 1', 'extra feature 2'], id_col='id')

def get_data_creator(backend='torch'):
    if False:
        for i in range(10):
            print('nop')
    if backend == 'torch':

        def data_creator(config):
            if False:
                print('Hello World!')
            import torch
            from torch.utils.data import TensorDataset, DataLoader
            tsdata = get_tsdataset()
            (x, y) = tsdata.roll(lookback=7, horizon=1).to_numpy()
            return DataLoader(TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).float()), batch_size=config['batch_size'], shuffle=True)
        return data_creator
    if backend == 'keras':

        def data_creator(config):
            if False:
                print('Hello World!')
            tsdata = get_tsdataset()
            tsdata.roll(lookback=7, horizon=1)
            return tsdata.to_tf_dataset(batch_size=config['batch_size'], shuffle=True)
        return data_creator

def gen_CustomizedNet():
    if False:
        i = 10
        return i + 15
    import torch
    import torch.nn as nn

    class CustomizedNet(nn.Module):

        def __init__(self, dropout, input_size, input_feature_num, hidden_dim, output_size):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Simply use linear layers for multi-variate single-step forecasting.\n            '
            super().__init__()
            self.fc1 = nn.Linear(input_size * input_feature_num, hidden_dim)
            self.dropout = nn.Dropout(dropout)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(hidden_dim, output_size)

        def forward(self, x):
            if False:
                print('Hello World!')
            x = x.view(-1, x.shape[1] * x.shape[2])
            x = self.fc1(x)
            x = self.dropout(x)
            x = self.relu1(x)
            x = self.fc2(x)
            x = torch.unsqueeze(x, 1)
            return x
    return CustomizedNet

def model_creator_pytorch(config):
    if False:
        print('Hello World!')
    '\n    Pytorch customized model creator\n    '
    CustomizedNet = gen_CustomizedNet()
    return CustomizedNet(dropout=config['dropout'], input_size=config['past_seq_len'], input_feature_num=config['input_feature_num'], hidden_dim=config['hidden_dim'], output_size=config['output_feature_num'])

def model_creator_keras(config):
    if False:
        print('Hello World!')
    '\n    Keras(tf2) customized model creator\n    '
    from bigdl.nano.tf.keras import Sequential
    model = Sequential([tf.keras.layers.Input(shape=(config['past_seq_len'], config['input_feature_num'])), tf.keras.layers.Dense(config['hidden_dim'], activation='relu'), tf.keras.layers.Dropout(config['dropout']), tf.keras.layers.Dense(config['output_feature_num'], activation='softmax')])
    learning_rate = config.get('lr', 0.001)
    optimizer = getattr(tf.keras.optimizers, config.get('optim', 'Adam'))(learning_rate)
    model.compile(loss=config.get('loss', 'mse'), optimizer=optimizer, metrics=[config.get('metric', 'mse')])
    return model

@op_distributed
class TestAutoTrainer(TestCase):

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        from bigdl.orca import init_orca_context
        init_orca_context(cores=8, init_ray_on_spark=True)

    def tearDown(self) -> None:
        if False:
            print('Hello World!')
        from bigdl.orca import stop_orca_context
        stop_orca_context()

    @op_torch
    def test_fit_third_party_feature(self):
        if False:
            print('Hello World!')
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        tsdata_train = get_tsdataset().gen_dt_feature().scale(scaler, fit=True)
        tsdata_valid = get_tsdataset().gen_dt_feature().scale(scaler, fit=False)
        search_space = {'hidden_dim': hp.grid_search([32, 64]), 'dropout': hp.uniform(0.1, 0.2)}
        auto_estimator = AutoTSEstimator(model=model_creator_pytorch, search_space=search_space, past_seq_len=hp.randint(4, 6), future_seq_len=1, selected_features='auto', metric='mse', loss=torch.nn.MSELoss(), cpus_per_trial=2)
        ts_pipeline = auto_estimator.fit(data=tsdata_train, epochs=1, batch_size=hp.choice([32, 64]), validation_data=tsdata_valid, n_sampling=1)
        best_config = auto_estimator.get_best_config()
        best_model = auto_estimator._get_best_automl_model()
        assert 4 <= best_config['past_seq_len'] <= 6
        from bigdl.chronos.autots.tspipeline import TSPipeline
        assert isinstance(ts_pipeline, TSPipeline)
        tsdata_valid.roll(lookback=best_config['past_seq_len'], horizon=0, feature_col=best_config['selected_features'])
        x_valid = tsdata_valid.to_numpy()
        y_pred_raw = best_model.predict(x_valid)
        y_pred_raw = tsdata_valid.unscale_numpy(y_pred_raw)
        eval_result = ts_pipeline.evaluate(tsdata_valid)
        y_pred = ts_pipeline.predict(tsdata_valid)
        np.testing.assert_almost_equal(y_pred, y_pred_raw)
        ts_pipeline.save('/tmp/auto_trainer/autots_tmp_model_3rdparty')
        new_ts_pipeline = TSPipeline.load('/tmp/auto_trainer/autots_tmp_model_3rdparty')
        eval_result_new = new_ts_pipeline.evaluate(tsdata_valid)
        y_pred_new = new_ts_pipeline.predict(tsdata_valid)
        np.testing.assert_almost_equal(eval_result[0], eval_result_new[0])
        np.testing.assert_almost_equal(y_pred, y_pred_new)
        new_ts_pipeline.fit(tsdata_valid)

    @op_tf2
    def test_fit_third_party_feature_tf2(self):
        if False:
            for i in range(10):
                print('nop')
        search_space = {'hidden_dim': hp.grid_search([32, 64]), 'layer_num': hp.randint(1, 3), 'dropout': hp.uniform(0.1, 0.2)}
        auto_estimator = AutoTSEstimator(model=model_creator_keras, search_space=search_space, past_seq_len=7, future_seq_len=1, input_feature_num=None, output_target_num=None, selected_features='auto', metric='mse', backend='keras', logs_dir='/tmp/auto_trainer', cpus_per_trial=2, name='auto_trainer')
        auto_estimator.fit(data=get_tsdataset(), epochs=1, batch_size=hp.choice([32, 64]), validation_data=get_tsdataset(), n_sampling=1)
        config = auto_estimator.get_best_config()
        assert config['past_seq_len'] == 7

    @op_torch
    def test_fit_third_party_data_creator(self):
        if False:
            i = 10
            return i + 15
        input_feature_dim = 4
        output_feature_dim = 2
        search_space = {'hidden_dim': hp.grid_search([32, 64]), 'dropout': hp.uniform(0.1, 0.2)}
        auto_estimator = AutoTSEstimator(model=model_creator_pytorch, search_space=search_space, past_seq_len=7, future_seq_len=1, input_feature_num=input_feature_dim, output_target_num=output_feature_dim, selected_features='auto', metric='mse', loss=torch.nn.MSELoss(), cpus_per_trial=2)
        auto_estimator.fit(data=get_data_creator(), epochs=1, batch_size=hp.choice([32, 64]), validation_data=get_data_creator(), n_sampling=1)
        config = auto_estimator.get_best_config()
        assert config['past_seq_len'] == 7

    @op_tf2
    def test_fit_third_party_data_creator_tf2(self):
        if False:
            return 10
        search_space = {'hidden_dim': hp.grid_search([32, 64]), 'layer_num': hp.randint(1, 3), 'dropout': hp.uniform(0.1, 0.2)}
        auto_estimator = AutoTSEstimator(model=model_creator_keras, search_space=search_space, past_seq_len=7, future_seq_len=1, input_feature_num=4, output_target_num=2, selected_features='auto', metric='mse', backend='keras', logs_dir='/tmp/auto_trainer', cpus_per_trial=2, name='auto_trainer')
        auto_estimator.fit(data=get_data_creator(backend='keras'), epochs=1, batch_size=hp.choice([32, 64]), validation_data=get_data_creator(backend='keras'), n_sampling=1)
        config = auto_estimator.get_best_config()
        assert config['past_seq_len'] == 7

    @op_torch
    def test_fit_customized_metrics(self):
        if False:
            while True:
                i = 10
        from sklearn.preprocessing import StandardScaler
        import torch
        from torchmetrics.functional import mean_squared_error
        import random
        scaler = StandardScaler()
        tsdata_train = get_tsdataset().gen_dt_feature().scale(scaler, fit=True)
        tsdata_valid = get_tsdataset().gen_dt_feature().scale(scaler, fit=False)

        def customized_metric(y_true, y_pred):
            if False:
                i = 10
                return i + 15
            return mean_squared_error(torch.from_numpy(y_pred), torch.from_numpy(y_true)).numpy()
        auto_estimator = AutoTSEstimator(model=random.choice(['tcn', 'lstm', 'seq2seq']), search_space='minimal', past_seq_len=hp.randint(4, 6), future_seq_len=1, selected_features='auto', metric=customized_metric, metric_mode='min', optimizer='Adam', loss=torch.nn.MSELoss(), logs_dir='/tmp/auto_trainer', cpus_per_trial=2, name='auto_trainer')
        ts_pipeline = auto_estimator.fit(data=tsdata_train, epochs=1, batch_size=hp.choice([32, 64]), validation_data=tsdata_valid, n_sampling=1)
        best_config = auto_estimator.get_best_config()
        best_model = auto_estimator._get_best_automl_model()
        assert 4 <= best_config['past_seq_len'] <= 6

    @op_torch
    @op_inference
    def test_fit_lstm_feature(self):
        if False:
            while True:
                i = 10
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        tsdata_train = get_tsdataset().gen_dt_feature().scale(scaler, fit=True)
        tsdata_valid = get_tsdataset().gen_dt_feature().scale(scaler, fit=False)
        auto_estimator = AutoTSEstimator(model='lstm', search_space='minimal', past_seq_len=hp.randint(4, 6), future_seq_len=1, selected_features='auto', metric='mse', loss=torch.nn.MSELoss(), logs_dir='/tmp/auto_trainer', cpus_per_trial=2, name='auto_trainer')
        ts_pipeline = auto_estimator.fit(data=tsdata_train, epochs=1, batch_size=hp.choice([32, 64]), validation_data=tsdata_valid, n_sampling=1)
        best_config = auto_estimator.get_best_config()
        best_model = auto_estimator._get_best_automl_model()
        assert 4 <= best_config['past_seq_len'] <= 6
        from bigdl.chronos.autots.tspipeline import TSPipeline
        assert isinstance(ts_pipeline, TSPipeline)
        tsdata_valid.roll(lookback=best_config['past_seq_len'], horizon=0, feature_col=best_config['selected_features'])
        x_valid = tsdata_valid.to_numpy()
        y_pred_raw = best_model.predict(x_valid)
        y_pred_raw = tsdata_valid.unscale_numpy(y_pred_raw)
        eval_result = ts_pipeline.evaluate(tsdata_valid)
        y_pred = ts_pipeline.predict(tsdata_valid)
        np.testing.assert_almost_equal(y_pred, y_pred_raw)
        ts_pipeline.save('/tmp/auto_trainer/autots_tmp_model_lstm')
        new_ts_pipeline = TSPipeline.load('/tmp/auto_trainer/autots_tmp_model_lstm')
        eval_result_new = new_ts_pipeline.evaluate(tsdata_valid)
        y_pred_new = new_ts_pipeline.predict(tsdata_valid)
        np.testing.assert_almost_equal(eval_result[0], eval_result_new[0])
        np.testing.assert_almost_equal(y_pred, y_pred_new)
        try:
            import onnx
            import onnxruntime
            eval_result_new_onnx = new_ts_pipeline.evaluate_with_onnx(tsdata_valid)
            y_pred_new_onnx = new_ts_pipeline.predict_with_onnx(tsdata_valid)
            np.testing.assert_almost_equal(eval_result[0], eval_result_new_onnx[0], decimal=5)
            np.testing.assert_almost_equal(y_pred, y_pred_new_onnx, decimal=5)
        except ImportError:
            pass
        new_ts_pipeline.fit(tsdata_valid)

    @op_torch
    @op_inference
    def test_fit_tcn_feature(self):
        if False:
            while True:
                i = 10
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        tsdata_train = get_tsdataset().gen_dt_feature().scale(scaler, fit=True)
        tsdata_valid = get_tsdataset().gen_dt_feature().scale(scaler, fit=False)
        auto_estimator = AutoTSEstimator(model='tcn', search_space='minimal', past_seq_len=hp.randint(4, 6), future_seq_len=1, selected_features='auto', metric='mse', optimizer='Adam', loss=torch.nn.MSELoss(), logs_dir='/tmp/auto_trainer', cpus_per_trial=2, name='auto_trainer')
        ts_pipeline = auto_estimator.fit(data=tsdata_train, epochs=1, batch_size=hp.choice([32, 64]), validation_data=tsdata_valid, n_sampling=1)
        best_config = auto_estimator.get_best_config()
        best_model = auto_estimator._get_best_automl_model()
        assert 4 <= best_config['past_seq_len'] <= 6
        from bigdl.chronos.autots.tspipeline import TSPipeline
        assert isinstance(ts_pipeline, TSPipeline)
        tsdata_valid.roll(lookback=best_config['past_seq_len'], horizon=0, feature_col=best_config['selected_features'])
        x_valid = tsdata_valid.to_numpy()
        y_pred_raw = best_model.predict(x_valid)
        y_pred_raw = tsdata_valid.unscale_numpy(y_pred_raw)
        eval_result = ts_pipeline.evaluate(tsdata_valid)
        y_pred = ts_pipeline.predict(tsdata_valid)
        np.testing.assert_almost_equal(y_pred, y_pred_raw)
        ts_pipeline.save('/tmp/auto_trainer/autots_tmp_model_tcn')
        new_ts_pipeline = TSPipeline.load('/tmp/auto_trainer/autots_tmp_model_tcn')
        eval_result_new = new_ts_pipeline.evaluate(tsdata_valid)
        y_pred_new = new_ts_pipeline.predict(tsdata_valid)
        np.testing.assert_almost_equal(eval_result[0], eval_result_new[0])
        np.testing.assert_almost_equal(y_pred, y_pred_new)
        try:
            import onnx
            import onnxruntime
            eval_result_new_onnx = new_ts_pipeline.evaluate_with_onnx(tsdata_valid)
            y_pred_new_onnx = new_ts_pipeline.predict_with_onnx(tsdata_valid)
            np.testing.assert_almost_equal(eval_result[0], eval_result_new_onnx[0], decimal=5)
            np.testing.assert_almost_equal(y_pred, y_pred_new_onnx, decimal=5)
        except ImportError:
            pass
        new_ts_pipeline.fit(tsdata_valid)

    @op_torch
    @op_inference
    def test_fit_seq2seq_feature(self):
        if False:
            for i in range(10):
                print('nop')
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        tsdata_train = get_tsdataset().gen_dt_feature().scale(scaler, fit=True)
        tsdata_valid = get_tsdataset().gen_dt_feature().scale(scaler, fit=False)
        auto_estimator = AutoTSEstimator(model='seq2seq', search_space='minimal', past_seq_len=hp.randint(4, 6), future_seq_len=1, selected_features='auto', metric='mse', optimizer='Adam', loss=torch.nn.MSELoss(), logs_dir='/tmp/auto_trainer', cpus_per_trial=2, name='auto_trainer')
        ts_pipeline = auto_estimator.fit(data=tsdata_train, epochs=1, batch_size=hp.choice([32, 64]), validation_data=tsdata_valid, n_sampling=1)
        best_config = auto_estimator.get_best_config()
        best_model = auto_estimator._get_best_automl_model()
        assert 4 <= best_config['past_seq_len'] <= 6
        from bigdl.chronos.autots.tspipeline import TSPipeline
        assert isinstance(ts_pipeline, TSPipeline)
        tsdata_valid.roll(lookback=best_config['past_seq_len'], horizon=0, feature_col=best_config['selected_features'])
        x_valid = tsdata_valid.to_numpy()
        y_pred_raw = best_model.predict(x_valid)
        y_pred_raw = tsdata_valid.unscale_numpy(y_pred_raw)
        eval_result = ts_pipeline.evaluate(tsdata_valid)
        y_pred = ts_pipeline.predict(tsdata_valid)
        np.testing.assert_almost_equal(y_pred, y_pred_raw)
        ts_pipeline.save('/tmp/auto_trainer/autots_tmp_model_seq2seq')
        new_ts_pipeline = TSPipeline.load('/tmp/auto_trainer/autots_tmp_model_seq2seq')
        eval_result_new = new_ts_pipeline.evaluate(tsdata_valid)
        y_pred_new = new_ts_pipeline.predict(tsdata_valid)
        np.testing.assert_almost_equal(eval_result[0], eval_result_new[0])
        np.testing.assert_almost_equal(y_pred, y_pred_new)
        try:
            import onnx
            import onnxruntime
            eval_result_new_onnx = new_ts_pipeline.evaluate_with_onnx(tsdata_valid)
            y_pred_new_onnx = new_ts_pipeline.predict_with_onnx(tsdata_valid)
            np.testing.assert_almost_equal(eval_result[0], eval_result_new_onnx[0], decimal=5)
            np.testing.assert_almost_equal(y_pred, y_pred_new_onnx, decimal=5)
        except ImportError:
            pass
        new_ts_pipeline.fit(tsdata_valid)

    @op_torch
    def test_fit_lstm_data_creator(self):
        if False:
            print('Hello World!')
        input_feature_dim = 4
        output_feature_dim = 2
        search_space = {'hidden_dim': hp.grid_search([32, 64]), 'layer_num': hp.randint(1, 3), 'lr': hp.choice([0.001, 0.003, 0.01]), 'dropout': hp.uniform(0.1, 0.2)}
        auto_estimator = AutoTSEstimator(model='lstm', search_space=search_space, past_seq_len=7, future_seq_len=1, input_feature_num=input_feature_dim, output_target_num=output_feature_dim, selected_features='auto', metric='mse', loss=torch.nn.MSELoss(), logs_dir='/tmp/auto_trainer', cpus_per_trial=2, name='auto_trainer')
        auto_estimator.fit(data=get_data_creator(), epochs=1, batch_size=hp.choice([32, 64]), validation_data=get_data_creator(), n_sampling=1)
        config = auto_estimator.get_best_config()
        assert config['past_seq_len'] == 7

    @op_torch
    def test_select_feature(self):
        if False:
            while True:
                i = 10
        sample_num = np.random.randint(100, 200)
        df = pd.DataFrame({'datetime': pd.date_range('1/1/2019', periods=sample_num), 'value': np.random.randn(sample_num), 'id': np.array(['00'] * sample_num)})
        (train_ts, val_ts, _) = TSDataset.from_pandas(df, target_col=['value'], dt_col='datetime', id_col='id', with_split=True, val_ratio=0.1)
        search_space = {'hidden_dim': hp.grid_search([32, 64]), 'layer_num': hp.randint(1, 3), 'lr': hp.choice([0.001, 0.003, 0.01]), 'dropout': hp.uniform(0.1, 0.2)}
        (input_feature_dim, output_feature_dim) = (1, 1)
        auto_estimator = AutoTSEstimator(model='lstm', search_space=search_space, past_seq_len=6, future_seq_len=1, input_feature_num=input_feature_dim, output_target_num=output_feature_dim, selected_features='auto', metric='mse', loss=torch.nn.MSELoss(), cpus_per_trial=2, name='auto_trainer')
        auto_estimator.fit(data=train_ts, epochs=1, batch_size=hp.choice([32, 64]), validation_data=val_ts, n_sampling=1)
        config = auto_estimator.get_best_config()
        assert config['past_seq_len'] == 6

    @op_torch
    def test_future_list_input(self):
        if False:
            i = 10
            return i + 15
        sample_num = np.random.randint(100, 200)
        df = pd.DataFrame({'datetime': pd.date_range('1/1/2019', periods=sample_num), 'value': np.random.randn(sample_num), 'id': np.array(['00'] * sample_num)})
        (train_ts, val_ts, _) = TSDataset.from_pandas(df, target_col=['value'], dt_col='datetime', id_col='id', with_split=True, val_ratio=0.1)
        (input_feature_dim, output_feature_dim) = (1, 1)
        auto_estimator = AutoTSEstimator(model='seq2seq', search_space='minimal', past_seq_len=6, future_seq_len=[1, 3], input_feature_num=input_feature_dim, output_target_num=output_feature_dim, selected_features='auto', metric='mse', loss=torch.nn.MSELoss(), cpus_per_trial=2, name='auto_trainer')
        auto_estimator.fit(data=train_ts, epochs=1, batch_size=hp.choice([32, 64]), validation_data=val_ts, n_sampling=1)
        config = auto_estimator.get_best_config()
        assert config['future_seq_len'] == 2
        assert auto_estimator._future_seq_len == [1, 3]

    @op_torch
    def test_autogener_best_cycle_length(self):
        if False:
            for i in range(10):
                print('nop')
        sample_num = 100
        df = pd.DataFrame({'datetime': pd.date_range('1/1/2019', periods=sample_num), 'value': np.sin(np.array((0, 30, 45, 60, 90) * 20) * np.pi / 180), 'id': np.array(['00'] * sample_num)})
        train_ts = TSDataset.from_pandas(df, target_col=['value'], dt_col='datetime', id_col='id', with_split=False)
        (input_feature_dim, output_feature_dim) = (1, 1)
        auto_estimator = AutoTSEstimator(model='lstm', search_space='minimal', past_seq_len='auto', input_feature_num=input_feature_dim, output_target_num=output_feature_dim)
        auto_estimator.fit(data=train_ts, epochs=1, batch_size=hp.choice([16, 32]), validation_data=train_ts)
        config = auto_estimator.get_best_config()
        assert 2 <= config['past_seq_len'] <= 10
if __name__ == '__main__':
    pytest.main([__file__])