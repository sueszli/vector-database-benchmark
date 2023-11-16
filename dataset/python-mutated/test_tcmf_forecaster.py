import pytest
import numpy as np
from bigdl.chronos.utils import LazyImport
TCMFForecaster = LazyImport('bigdl.chronos.forecaster.tcmf_forecaster.TCMFForecaster')
from unittest import TestCase
import tempfile
import pandas as pd
from .. import op_distributed, op_torch, op_diff_set_all

@op_torch
@op_distributed
class TestChronosModelTCMFForecaster(TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.model = TCMFForecaster()
        self.num_samples = 300
        self.horizon = np.random.randint(1, 50)
        self.seq_len = 480
        self.data = np.random.rand(self.num_samples, self.seq_len)
        self.id = np.arange(self.num_samples)
        self.data_new = np.random.rand(self.num_samples, self.horizon)
        self.fit_params = dict(val_len=12, start_date='2020-1-1', freq='5min', y_iters=1, init_FX_epoch=1, max_FX_epoch=1, max_TCN_epoch=1, alt_iters=2)

    def tearDown(self):
        if False:
            return 10
        pass

    @classmethod
    def tearDownClass(cls):
        if False:
            print('Hello World!')
        from pyspark import SparkContext
        from bigdl.orca.ray import OrcaRayContext
        if SparkContext._active_spark_context is not None:
            print('Stopping spark_orca context')
            sc = SparkContext.getOrCreate()
            if sc.getConf().get('spark.master').startswith('spark://'):
                from bigdl.dllib.nncontext import stop_spark_standalone
                stop_spark_standalone()
            sc.stop()
        if OrcaRayContext._active_ray_context is not None:
            print('Stopping ray_orca context')
            ray_ctx = OrcaRayContext.get(initialize=False)
            if ray_ctx.initialized:
                ray_ctx.stop()

    def test_forecast_tcmf_ndarray(self):
        if False:
            for i in range(10):
                print('nop')
        ndarray_input = {'id': self.id, 'y': self.data}
        self.model.fit(ndarray_input, **self.fit_params)
        assert not self.model.is_xshards_distributed()
        yhat = self.model.predict(horizon=self.horizon)
        with tempfile.TemporaryDirectory() as tempdirname:
            self.model.save(tempdirname)
            loaded_model = TCMFForecaster.load(tempdirname, is_xshards_distributed=False)
        yhat_loaded = loaded_model.predict(horizon=self.horizon)
        yhat_id = yhat_loaded['id']
        np.testing.assert_equal(yhat_id, self.id)
        yhat = yhat['prediction']
        yhat_loaded = yhat_loaded['prediction']
        assert yhat.shape == (self.num_samples, self.horizon)
        np.testing.assert_array_almost_equal(yhat, yhat_loaded, decimal=4)
        target_value = dict({'y': self.data_new})
        assert self.model.evaluate(target_value=target_value, metric=['mse'])
        self.model.fit_incremental({'y': self.data_new})
        self.model.fit_incremental({'y': self.data_new})
        yhat_incr = self.model.predict(horizon=self.horizon)
        yhat_incr = yhat_incr['prediction']
        assert yhat_incr.shape == (self.num_samples, self.horizon)
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, yhat, yhat_incr)

    def test_tcmf_ndarray_covariates_dti(self):
        if False:
            i = 10
            return i + 15
        ndarray_input = {'id': self.id, 'y': self.data}
        self.model.fit(ndarray_input, covariates=np.random.rand(3, self.seq_len), dti=pd.date_range('20130101', periods=self.seq_len), **self.fit_params)
        future_covariates = np.random.randn(3, self.horizon)
        future_dti = pd.date_range('20130101', periods=self.horizon)
        yhat = self.model.predict(horizon=self.horizon, future_covariates=future_covariates, future_dti=future_dti)
        with tempfile.TemporaryDirectory() as tempdirname:
            self.model.save(tempdirname)
            loaded_model = TCMFForecaster.load(tempdirname, is_xshards_distributed=False)
        yhat_loaded = loaded_model.predict(horizon=self.horizon, future_covariates=future_covariates, future_dti=future_dti)
        yhat_id = yhat_loaded['id']
        np.testing.assert_equal(yhat_id, self.id)
        yhat = yhat['prediction']
        yhat_loaded = yhat_loaded['prediction']
        assert yhat.shape == (self.num_samples, self.horizon)
        np.testing.assert_array_almost_equal(yhat, yhat_loaded, decimal=4)
        target_value = dict({'y': self.data_new})
        assert self.model.evaluate(target_value=target_value, target_covariates=future_covariates, target_dti=future_dti, metric=['mse'])
        self.model.fit_incremental({'y': self.data_new}, covariates_incr=future_covariates, dti_incr=future_dti)
        yhat_incr = self.model.predict(horizon=self.horizon, future_covariates=future_covariates, future_dti=future_dti)
        yhat_incr = yhat_incr['prediction']
        assert yhat_incr.shape == (self.num_samples, self.horizon)
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, yhat, yhat_incr)

    def test_forecast_ndarray_error(self):
        if False:
            return 10
        with self.assertRaises(Exception) as context:
            self.model.is_xshards_distributed()
        print(str(context.exception))
        self.assertTrue('You should run fit before calling is_xshards_distributed()' in str(context.exception))
        input = dict({'data': self.data})
        with self.assertRaises(Exception) as context:
            self.model.fit(input)
        self.assertTrue("key `y` doesn't exist in x" in str(context.exception))
        input = dict({'y': 'abc'})
        with self.assertRaises(Exception) as context:
            self.model.fit(input)
        self.assertTrue('the value of y should be an ndarray' in str(context.exception))
        id_diff = np.arange(200)
        input = dict({'id': id_diff, 'y': self.data})
        with self.assertRaises(Exception) as context:
            self.model.fit(input)
        self.assertTrue('the length of the id array should be equal to the number of' in str(context.exception))
        input_right = dict({'id': self.id, 'y': self.data})
        self.model.fit(input_right, **self.fit_params)
        with self.assertRaises(Exception) as context:
            self.model.fit(input_right)
        self.assertTrue('This model has already been fully trained' in str(context.exception))
        data_id_diff = {'id': self.id - 1, 'y': self.data_new}
        with self.assertRaises(RuntimeError) as context:
            self.model.fit_incremental(data_id_diff)
        self.assertTrue('The input ids in fit_incremental differs from input ids in fit' in str(context.exception))
        target_value_fake = dict({'data': self.data_new})
        with self.assertRaises(Exception) as context:
            self.model.evaluate(target_value=target_value_fake, metric=['mse'])
        self.assertTrue("key `y` doesn't exist in x" in str(context.exception))

    def test_forecast_tcmf_without_id(self):
        if False:
            i = 10
            return i + 15
        input = dict({'y': self.data})
        self.model.fit(input, **self.fit_params)
        assert not self.model.is_xshards_distributed()
        with tempfile.TemporaryDirectory() as tempdirname:
            self.model.save(tempdirname)
            loaded_model = TCMFForecaster.load(tempdirname, is_xshards_distributed=False)
        yhat = self.model.predict(horizon=self.horizon)
        yhat_loaded = loaded_model.predict(horizon=self.horizon)
        assert 'id' not in yhat_loaded
        yhat = yhat['prediction']
        yhat_loaded = yhat_loaded['prediction']
        assert yhat.shape == (self.num_samples, self.horizon)
        np.testing.assert_array_almost_equal(yhat, yhat_loaded, decimal=4)
        target_value = dict({'y': self.data_new})
        self.model.evaluate(target_value=target_value, metric=['mse'])
        self.model.fit_incremental({'y': self.data_new})
        yhat_incr = self.model.predict(horizon=self.horizon)
        yhat_incr = yhat_incr['prediction']
        assert yhat_incr.shape == (self.num_samples, self.horizon)
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, yhat, yhat_incr)
        data_new_id = {'id': self.id, 'y': self.data_new}
        with self.assertRaises(RuntimeError) as context:
            self.model.fit_incremental(data_new_id)
        self.assertTrue('Got valid id in fit_incremental and invalid id in fit.' in str(context.exception))

    def test_forecast_tcmf_xshards(self):
        if False:
            i = 10
            return i + 15
        from bigdl.orca import OrcaContext
        import bigdl.orca.data.pandas
        import pandas as pd
        OrcaContext.pandas_read_backend = 'pandas'

        def preprocessing(df, id_name, y_name):
            if False:
                i = 10
                return i + 15
            id = df.index
            data = df.to_numpy()
            result = dict({id_name: id, y_name: data})
            return result

        def postprocessing(pred_results, output_dt_col_name):
            if False:
                return 10
            id_arr = pred_results['id']
            pred_results = pred_results['prediction']
            pred_results = np.concatenate((np.expand_dims(id_arr, axis=1), pred_results), axis=1)
            final_df = pd.DataFrame(pred_results, columns=['id'] + output_dt_col_name)
            final_df.id = final_df.id.astype('int')
            final_df = final_df.set_index('id')
            final_df.columns.name = 'datetime'
            final_df = final_df.unstack().reset_index().rename({0: 'prediction'}, axis=1)
            return final_df

        def get_pred(d):
            if False:
                while True:
                    i = 10
            return d['prediction']
        with tempfile.NamedTemporaryFile() as temp:
            data = np.random.rand(300, 480)
            df = pd.DataFrame(data)
            df.to_csv(temp.name)
            shard = bigdl.orca.data.pandas.read_csv(temp.name)
        shard.cache()
        shard_train = shard.transform_shard(preprocessing, 'id', 'data')
        with self.assertRaises(Exception) as context:
            self.model.fit(shard_train)
        self.assertTrue("key `y` doesn't exist in x" in str(context.exception))
        shard_train = shard.transform_shard(preprocessing, 'cid', 'y')
        with self.assertRaises(Exception) as context:
            self.model.fit(shard_train)
        self.assertTrue("key `id` doesn't exist in x" in str(context.exception))
        with self.assertRaises(Exception) as context:
            self.model.is_xshards_distributed()
        self.assertTrue('You should run fit before calling is_xshards_distributed()' in str(context.exception))
        shard_train = shard.transform_shard(preprocessing, 'id', 'y')
        self.model.fit(shard_train, **self.fit_params)
        assert self.model.is_xshards_distributed()
        with self.assertRaises(Exception) as context:
            self.model.fit(shard_train)
        self.assertTrue('This model has already been fully trained' in str(context.exception))
        with self.assertRaises(Exception) as context:
            self.model.fit_incremental(shard_train)
        self.assertTrue('Error' in context.exception.__class__.__name__)
        with tempfile.TemporaryDirectory() as tempdirname:
            self.model.save(tempdirname + '/model')
            loaded_model = TCMFForecaster.load(tempdirname + '/model', is_xshards_distributed=True)
        horizon = np.random.randint(1, 50)
        yhat_shard_origin = self.model.predict(horizon=horizon)
        yhat_list_origin = yhat_shard_origin.collect()
        yhat_list_origin = list(map(get_pred, yhat_list_origin))
        yhat_shard = loaded_model.predict(horizon=horizon)
        yhat_list = yhat_shard.collect()
        yhat_list = list(map(get_pred, yhat_list))
        yhat_origin = np.concatenate(yhat_list_origin)
        yhat = np.concatenate(yhat_list)
        assert yhat.shape == (300, horizon)
        np.testing.assert_equal(yhat, yhat_origin)
        output_dt_col_name = pd.date_range(start='2020-05-01', periods=horizon, freq='H').to_list()
        yhat_df_shards = yhat_shard.transform_shard(postprocessing, output_dt_col_name)
        final_df_list = yhat_df_shards.collect()
        final_df = pd.concat(final_df_list)
        final_df.sort_values('datetime', inplace=True)
        assert final_df.shape == (300 * horizon, 3)
        OrcaContext.pandas_read_backend = 'spark'

    @op_diff_set_all
    def test_forecast_tcmf_distributed(self):
        if False:
            i = 10
            return i + 15
        input = dict({'id': self.id, 'y': self.data})
        from bigdl.orca import init_orca_context, stop_orca_context
        init_orca_context(cores=4, spark_log_level='INFO', init_ray_on_spark=True, object_store_memory='1g')
        self.model.fit(input, num_workers=4, **self.fit_params)
        with tempfile.TemporaryDirectory() as tempdirname:
            self.model.save(tempdirname)
            loaded_model = TCMFForecaster.load(tempdirname, is_xshards_distributed=False)
        yhat = self.model.predict(horizon=self.horizon, num_workers=4)
        yhat_loaded = loaded_model.predict(horizon=self.horizon, num_workers=4)
        yhat_id = yhat_loaded['id']
        np.testing.assert_equal(yhat_id, self.id)
        yhat = yhat['prediction']
        yhat_loaded = yhat_loaded['prediction']
        assert yhat.shape == (self.num_samples, self.horizon)
        np.testing.assert_equal(yhat, yhat_loaded)
        self.model.fit_incremental({'y': self.data_new})
        yhat_incr = self.model.predict(horizon=self.horizon)
        yhat_incr = yhat_incr['prediction']
        assert yhat_incr.shape == (self.num_samples, self.horizon)
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, yhat, yhat_incr)
        target_value = dict({'y': self.data_new})
        assert self.model.evaluate(target_value=target_value, metric=['mse'])
        stop_orca_context()
if __name__ == '__main__':
    pytest.main([__file__])