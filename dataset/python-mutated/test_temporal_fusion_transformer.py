import pickle
import shutil
import sys
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pytest
import torch
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.data.encoders import GroupNormalizer, MultiNormalizer
from pytorch_forecasting.metrics import CrossEntropy, MQF2DistributionLoss, MultiLoss, NegativeBinomialDistributionLoss, PoissonLoss, QuantileLoss, TweedieLoss
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
if sys.version.startswith('3.6'):
    from contextlib import contextmanager

    @contextmanager
    def nullcontext(enter_result=None):
        if False:
            while True:
                i = 10
        yield enter_result
else:
    from contextlib import nullcontext
from test_models.conftest import make_dataloaders

def test_integration(multiple_dataloaders_with_covariates, tmp_path):
    if False:
        return 10
    _integration(multiple_dataloaders_with_covariates, tmp_path, trainer_kwargs=dict(accelerator='cpu'))

def test_non_causal_attention(dataloaders_with_covariates, tmp_path):
    if False:
        return 10
    _integration(dataloaders_with_covariates, tmp_path, causal_attention=False, loss=TweedieLoss(), trainer_kwargs=dict(accelerator='cpu'))

def test_distribution_loss(data_with_covariates, tmp_path):
    if False:
        print('Hello World!')
    data_with_covariates = data_with_covariates.assign(volume=lambda x: x.volume.round())
    dataloaders_with_covariates = make_dataloaders(data_with_covariates, target='volume', time_varying_known_reals=['price_actual'], time_varying_unknown_reals=['volume'], static_categoricals=['agency'], add_relative_time_idx=True, target_normalizer=GroupNormalizer(groups=['agency', 'sku'], center=False))
    _integration(dataloaders_with_covariates, tmp_path, loss=NegativeBinomialDistributionLoss())

def test_mqf2_loss(data_with_covariates, tmp_path):
    if False:
        return 10
    data_with_covariates = data_with_covariates.assign(volume=lambda x: x.volume.round())
    dataloaders_with_covariates = make_dataloaders(data_with_covariates, target='volume', time_varying_known_reals=['price_actual'], time_varying_unknown_reals=['volume'], static_categoricals=['agency'], add_relative_time_idx=True, target_normalizer=GroupNormalizer(groups=['agency', 'sku'], center=False, transformation='log1p'))
    prediction_length = dataloaders_with_covariates['train'].dataset.min_prediction_length
    _integration(dataloaders_with_covariates, tmp_path, loss=MQF2DistributionLoss(prediction_length=prediction_length), learning_rate=0.001, trainer_kwargs=dict(accelerator='cpu'))

def _integration(dataloader, tmp_path, loss=None, trainer_kwargs=None, **kwargs):
    if False:
        while True:
            i = 10
    train_dataloader = dataloader['train']
    val_dataloader = dataloader['val']
    test_dataloader = dataloader['test']
    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=1, verbose=False, mode='min')
    logger = TensorBoardLogger(tmp_path)
    if trainer_kwargs is None:
        trainer_kwargs = {}
    trainer = pl.Trainer(max_epochs=2, gradient_clip_val=0.1, callbacks=[early_stop_callback], enable_checkpointing=True, default_root_dir=tmp_path, limit_train_batches=2, limit_val_batches=2, limit_test_batches=2, logger=logger, **trainer_kwargs)
    if 'discount_in_percent' in train_dataloader.dataset.reals:
        monotone_constaints = {'discount_in_percent': +1}
        cuda_context = torch.backends.cudnn.flags(enabled=False)
    else:
        monotone_constaints = {}
        cuda_context = nullcontext()
    kwargs.setdefault('learning_rate', 0.15)
    with cuda_context:
        if loss is not None:
            pass
        elif isinstance(train_dataloader.dataset.target_normalizer, NaNLabelEncoder):
            loss = CrossEntropy()
        elif isinstance(train_dataloader.dataset.target_normalizer, MultiNormalizer):
            loss = MultiLoss([CrossEntropy() if isinstance(normalizer, NaNLabelEncoder) else QuantileLoss() for normalizer in train_dataloader.dataset.target_normalizer.normalizers])
        else:
            loss = QuantileLoss()
        net = TemporalFusionTransformer.from_dataset(train_dataloader.dataset, hidden_size=2, hidden_continuous_size=2, attention_head_size=1, dropout=0.2, loss=loss, log_interval=5, log_val_interval=1, log_gradient_flow=True, monotone_constaints=monotone_constaints, **kwargs)
        net.size()
        try:
            trainer.fit(net, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
            if not isinstance(net.loss, MQF2DistributionLoss):
                test_outputs = trainer.test(net, dataloaders=test_dataloader)
                assert len(test_outputs) > 0
            net = TemporalFusionTransformer.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
            predictions = net.predict(val_dataloader, return_index=True, return_x=True, return_y=True, fast_dev_run=True, trainer_kwargs=trainer_kwargs)
            pred_len = len(predictions.index)

            def check(x):
                if False:
                    print('Hello World!')
                if isinstance(x, (tuple, list)):
                    for xi in x:
                        check(xi)
                elif isinstance(x, dict):
                    for xi in x.values():
                        check(xi)
                else:
                    assert pred_len == x.shape[0], 'first dimension should be prediction length'
            check(predictions.output)
            if isinstance(predictions.output, torch.Tensor):
                assert predictions.output.ndim == 2, 'shape of predictions should be batch_size x timesteps'
            else:
                assert all((p.ndim == 2 for p in predictions.output)), 'shape of predictions should be batch_size x timesteps'
            check(predictions.x)
            check(predictions.index)
            net.predict(val_dataloader, return_index=True, return_x=True, fast_dev_run=True, mode='raw', trainer_kwargs=trainer_kwargs)
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)

@pytest.fixture
def model(dataloaders_with_covariates):
    if False:
        print('Hello World!')
    dataset = dataloaders_with_covariates['train'].dataset
    net = TemporalFusionTransformer.from_dataset(dataset, learning_rate=0.15, hidden_size=4, attention_head_size=1, dropout=0.2, hidden_continuous_size=2, loss=PoissonLoss(), output_size=1, log_interval=5, log_val_interval=1, log_gradient_flow=True)
    return net

def test_tensorboard_graph_log(dataloaders_with_covariates, model, tmp_path):
    if False:
        i = 10
        return i + 15
    d = next(iter(dataloaders_with_covariates['train']))
    logger = TensorBoardLogger('test', str(tmp_path), log_graph=True)
    logger.log_graph(model, d[0])

def test_init_shared_network(dataloaders_with_covariates):
    if False:
        while True:
            i = 10
    dataset = dataloaders_with_covariates['train'].dataset
    net = TemporalFusionTransformer.from_dataset(dataset, share_single_variable_networks=True)
    net.predict(dataset, fast_dev_run=True)

@pytest.mark.parametrize('strategy', ['ddp'])
def test_distribution(dataloaders_with_covariates, tmp_path, strategy):
    if False:
        print('Hello World!')
    train_dataloader = dataloaders_with_covariates['train']
    val_dataloader = dataloaders_with_covariates['val']
    net = TemporalFusionTransformer.from_dataset(train_dataloader.dataset)
    logger = TensorBoardLogger(tmp_path)
    trainer = pl.Trainer(max_epochs=3, gradient_clip_val=0.1, fast_dev_run=True, logger=logger, strategy=strategy, enable_checkpointing=True, accelerator='gpu' if torch.cuda.is_available() else 'cpu')
    try:
        trainer.fit(net, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)

def test_pickle(model):
    if False:
        return 10
    pkl = pickle.dumps(model)
    pickle.loads(pkl)

@pytest.mark.parametrize('kwargs', [dict(mode='dataframe'), dict(mode='series'), dict(mode='raw')])
def test_predict_dependency(model, dataloaders_with_covariates, data_with_covariates, kwargs):
    if False:
        for i in range(10):
            print('nop')
    train_dataset = dataloaders_with_covariates['train'].dataset
    data_with_covariates = data_with_covariates.copy()
    dataset = TimeSeriesDataSet.from_dataset(train_dataset, data_with_covariates[lambda x: x.agency == data_with_covariates.agency.iloc[0]], predict=True)
    model.predict_dependency(dataset, variable='discount', values=[0.1, 0.0], **kwargs)
    model.predict_dependency(dataset, variable='agency', values=data_with_covariates.agency.unique()[:2], **kwargs)

def test_actual_vs_predicted_plot(model, dataloaders_with_covariates):
    if False:
        for i in range(10):
            print('nop')
    prediction = model.predict(dataloaders_with_covariates['val'], return_x=True)
    averages = model.calculate_prediction_actual_by_variable(prediction.x, prediction.output)
    model.plot_prediction_actual_by_variable(averages)

@pytest.mark.parametrize('kwargs', [dict(mode='raw'), dict(mode='quantiles'), dict(return_index=True), dict(return_decoder_lengths=True), dict(return_x=True), dict(return_y=True)])
def test_prediction_with_dataloder(model, dataloaders_with_covariates, kwargs):
    if False:
        for i in range(10):
            print('nop')
    val_dataloader = dataloaders_with_covariates['val']
    model.predict(val_dataloader, fast_dev_run=True, **kwargs)

def test_prediction_with_dataloder_raw(data_with_covariates, tmp_path):
    if False:
        i = 10
        return i + 15
    test_data = data_with_covariates.copy()
    np.random.seed(2)
    test_data = test_data.sample(frac=0.5)
    dataset = TimeSeriesDataSet(test_data, time_idx='time_idx', max_encoder_length=8, max_prediction_length=10, min_prediction_length=1, min_encoder_length=1, target='volume', group_ids=['agency', 'sku'], constant_fill_strategy=dict(volume=0.0), allow_missing_timesteps=True, time_varying_unknown_reals=['volume'], time_varying_known_reals=['time_idx'], target_normalizer=GroupNormalizer(groups=['agency', 'sku']))
    net = TemporalFusionTransformer.from_dataset(dataset, learning_rate=1e-06, hidden_size=4, attention_head_size=1, dropout=0.2, hidden_continuous_size=2, log_interval=1, log_val_interval=1, log_gradient_flow=True)
    logger = TensorBoardLogger(tmp_path)
    trainer = pl.Trainer(max_epochs=1, gradient_clip_val=1e-06, logger=logger)
    trainer.fit(net, train_dataloaders=dataset.to_dataloader(batch_size=4, num_workers=0))
    res = net.predict(dataset.to_dataloader(batch_size=2, num_workers=0), mode='raw')
    net.interpret_output(res)['attention']
    assert net.interpret_output(res.iget(slice(1)))['attention'].size() == torch.Size((1, net.hparams.max_encoder_length))

def test_prediction_with_dataset(model, dataloaders_with_covariates):
    if False:
        while True:
            i = 10
    val_dataloader = dataloaders_with_covariates['val']
    model.predict(val_dataloader.dataset, fast_dev_run=True)

def test_prediction_with_write_to_disk(model, dataloaders_with_covariates, tmp_path):
    if False:
        return 10
    val_dataloader = dataloaders_with_covariates['val']
    res = model.predict(val_dataloader.dataset, fast_dev_run=True, output_dir=tmp_path)
    assert res is None, 'result should be empty when writing to disk'

def test_prediction_with_dataframe(model, data_with_covariates):
    if False:
        return 10
    model.predict(data_with_covariates, fast_dev_run=True)

@pytest.mark.parametrize('use_learning_rate_finder', [True, False])
def test_hyperparameter_optimization_integration(dataloaders_with_covariates, tmp_path, use_learning_rate_finder):
    if False:
        while True:
            i = 10
    train_dataloader = dataloaders_with_covariates['train']
    val_dataloader = dataloaders_with_covariates['val']
    try:
        optimize_hyperparameters(train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, model_path=tmp_path, max_epochs=1, n_trials=3, log_dir=tmp_path, trainer_kwargs=dict(fast_dev_run=True, limit_train_batches=3, enable_progress_bar=False), use_learning_rate_finder=use_learning_rate_finder, learning_rate_range=[1e-06, 0.01])
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)