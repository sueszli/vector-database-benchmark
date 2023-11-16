import torch
import numpy as np
from pandas import Timedelta
from bigdl.chronos.forecaster.abstract import Forecaster
from bigdl.chronos.forecaster.utils import read_csv, delete_folder
from bigdl.chronos.metric.forecast_metrics import Evaluator
from bigdl.chronos.model.autoformer import model_creator, loss_creator
from torch.utils.data import TensorDataset, DataLoader
from bigdl.chronos.model.autoformer.Autoformer import AutoFormer, _transform_config_to_namedtuple
from bigdl.nano.utils.common import invalidInputError, invalidOperationError
from bigdl.chronos.forecaster.utils import check_transformer_data
from bigdl.chronos.pytorch import TSTrainer as Trainer
from bigdl.chronos.data import TSDataset
from bigdl.nano.automl.hpo.space import Space
from bigdl.chronos.forecaster.utils_hpo import GenericTSTransformerLightningModule, _config_has_search_space
from bigdl.chronos.pytorch.context_manager import DummyForecasterContextManager, ForecasterContextManager
from .utils_hpo import _format_metric_str
import warnings
from tempfile import TemporaryDirectory
import os

class AutoformerForecaster(Forecaster):

    def __init__(self, past_seq_len, future_seq_len, input_feature_num, output_feature_num, freq, label_len=None, output_attention=False, moving_avg=25, d_model=128, embed='timeF', dropout=0.05, factor=3, n_head=8, d_ff=256, activation='gelu', e_layers=2, d_layers=1, optimizer='Adam', loss='mse', lr=0.0001, lr_scheduler_milestones=[3, 4, 5, 6, 7, 8, 9, 10], metrics=['mse'], seed=None, distributed=False, workers_per_node=1, distributed_backend='ray'):
        if False:
            return 10
        '\n        Build a AutoformerForecaster Forecast Model.\n\n        :param past_seq_len: Specify the history time steps (i.e. lookback).\n        :param future_seq_len: Specify the output time steps (i.e. horizon).\n        :param input_feature_num: Specify the feature dimension.\n        :param output_feature_num: Specify the output dimension.\n        :param freq: Freq for time features encoding. You may choose from "s",\n               "t","h","d","w","m" for second, minute, hour, day, week or month.\n        :param label_len: Start token length of AutoFormer decoder.\n        :param optimizer: Specify the optimizer used for training. This value\n               defaults to "Adam".\n        :param loss: str or pytorch loss instance, Specify the loss function\n               used for training. This value defaults to "mse". You can choose\n               from "mse", "mae", "huber_loss" or any customized loss instance\n               you want to use.\n        :param lr: Specify the learning rate. This value defaults to 0.001.\n        :param lr_scheduler_milestones: Specify the milestones parameters in\n               torch.optim.lr_scheduler.MultiStepLR.This value defaults to\n               [3, 4, 5, 6, 7, 8, 9, 10]. If you don\'t want to use scheduler,\n               set this parameter to None to disbale lr_scheduler.\n        :param metrics: A list contains metrics for evaluating the quality of\n               forecasting. You may only choose from "mse" and "mae" for a\n               distributed forecaster. You may choose from "mse", "mae",\n               "rmse", "r2", "mape", "smape" or a callable function for a\n               non-distributed forecaster. If callable function, it signature\n               should be func(y_true, y_pred), where y_true and y_pred are numpy\n               ndarray.\n        :param seed: int, random seed for training. This value defaults to None.\n        :param distributed: bool, if init the forecaster in a distributed\n               fashion. If True, the internal model will use an Orca Estimator.\n               If False, the internal model will use a pytorch model. The value\n               defaults to False.\n        :param workers_per_node: int, the number of worker you want to use.\n               The value defaults to 1. The param is only effective when\n               distributed is set to True.\n        :param distributed_backend: str, select from "ray" or\n               "horovod". The value defaults to "ray".\n        :param kwargs: other hyperparameter please refer to\n               https://github.com/zhouhaoyi/Informer2020#usage\n        '
        invalidInputError(past_seq_len > 1, 'past_seq_len of Autoformer must exceeds one.')
        self.data_config = {'past_seq_len': past_seq_len, 'future_seq_len': future_seq_len, 'input_feature_num': input_feature_num, 'output_feature_num': output_feature_num, 'label_len': past_seq_len // 2 if label_len is None else label_len}
        self.model_config = {'seq_len': past_seq_len, 'label_len': past_seq_len // 2 if label_len is None else label_len, 'pred_len': future_seq_len, 'output_attention': output_attention, 'moving_avg': moving_avg, 'enc_in': input_feature_num, 'd_model': d_model, 'embed': embed, 'freq': freq, 'dropout': dropout, 'dec_in': input_feature_num, 'factor': factor, 'n_head': n_head, 'd_ff': d_ff, 'activation': activation, 'e_layers': e_layers, 'c_out': output_feature_num, 'd_layers': d_layers, 'seed': seed}
        self.loss_config = {'loss': loss}
        self.optim_config = {'lr': lr, 'optim': optimizer, 'lr_scheduler_milestones': lr_scheduler_milestones}
        self.model_config.update(self.loss_config)
        self.model_config.update(self.optim_config)
        self.metrics = metrics
        self.distributed = distributed
        self.checkpoint_callback = True
        if not isinstance(seed, Space):
            from pytorch_lightning import seed_everything
            seed_everything(seed=seed, workers=True)
        self.num_processes = 1
        self.use_ipex = False
        self.onnx_available = False
        self.quantize_available = False
        self.use_amp = False
        self.use_hpo = True
        self.fitted = False
        has_space = _config_has_search_space(config={**self.model_config, **self.optim_config, **self.loss_config, **self.data_config})
        if not has_space:
            self.use_hpo = False
            self.internal = model_creator(self.model_config)
        self.model_creator = model_creator
        self.loss_creator = loss_creator
        self.cxt_manager = DummyForecasterContextManager()
        self.context_enabled = False
        current_num_threads = torch.get_num_threads()
        self.thread_num = current_num_threads
        self.accelerate_method = None

    def _build_automodel(self, data, validation_data=None, batch_size=32, epochs=1):
        if False:
            for i in range(10):
                print('nop')
        'Build a Generic Model using config parameters.'
        merged_config = {**self.model_config, **self.optim_config, **self.loss_config, **self.data_config}
        model_config_keys = list(self.model_config.keys())
        data_config_keys = list(self.data_config.keys())
        optim_config_keys = list(self.optim_config.keys())
        loss_config_keys = list(self.loss_config.keys())
        return GenericTSTransformerLightningModule(model_creator=self.model_creator, loss_creator=self.loss_creator, data=data, validation_data=validation_data, batch_size=batch_size, epochs=epochs, metrics=[_str2metric(metric) for metric in self.metrics], scheduler=None, num_processes=self.num_processes, model_config_keys=model_config_keys, data_config_keys=data_config_keys, optim_config_keys=optim_config_keys, loss_config_keys=loss_config_keys, **merged_config)

    def tune(self, data, validation_data, target_metric='mse', direction='minimize', directions=None, n_trials=2, n_parallels=1, epochs=1, batch_size=32, acceleration=False, input_sample=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Search the hyper parameter.\n\n        :param data: The data support following formats:\n\n               | 1. numpy ndarrays: generate from `TSDataset.roll`,\n                    be sure to set label_len > 0 and time_enc = True\n\n        :param validation_data: validation data, The data support following formats:\n\n               | 1. numpy ndarrays: generate from `TSDataset.roll`,\n                    be sure to set label_len > 0 and time_enc = True\n\n        :param target_metric: the target metric to optimize,\n               a string or an instance of torchmetrics.metric.Metric, default to \'mse\'.\n        :param direction: in which direction to optimize the target metric,\n               "maximize" - larger the better\n               "minimize" - smaller the better\n               default to "minimize".\n        :param n_trials: number of trials to run\n        :param n_parallels: number of parallel processes used to run trials.\n               to use parallel tuning you need to use a RDB url for storage and specify study_name.\n               For more information, refer to Nano AutoML user guide.\n        :param epochs: the number of epochs to run in each trial fit, defaults to 1\n        :param batch_size: number of batch size for each trial fit, defaults to 32\n        :param acceleration: Whether to automatically consider the model after\n            inference acceleration in the search process. It will only take\n            effect if target_metric contains "latency". Default value is False.\n        :param input_sample: A set of inputs for trace, defaults to None if you have\n            trace before or model is a LightningModule with any dataloader attached.\n        '
        invalidInputError(not self.distributed, 'HPO is not supported in distributed mode.Please use AutoTS instead.')
        invalidOperationError(self.use_hpo, 'HPO is disabled for this forecaster.You may specify search space in hyper parameters to enable it.')
        from bigdl.chronos.pytorch import TSTrainer as Trainer
        if isinstance(data, tuple):
            check_transformer_data(data[0], data[1], data[2], data[3], self.data_config)
            if validation_data and isinstance(validation_data, tuple):
                check_transformer_data(validation_data[0], validation_data[1], validation_data[2], validation_data[3], self.data_config)
            else:
                invalidInputError(False, 'To use tuning, you must provide validation_dataas numpy arrays.')
        else:
            invalidInputError(False, 'HPO only supports numpy train input data.')
        if input_sample is None:
            input_sample = (torch.from_numpy(data[0][:1, :, :]), torch.from_numpy(data[1][:1, :, :]), torch.from_numpy(data[2][:1, :, :]), torch.from_numpy(data[3][:1, :, :]))
        if validation_data is not None:
            formated_target_metric = _format_metric_str('val', target_metric)
        else:
            invalidInputError(False, 'To use tuning, you must provide validation_dataas numpy arrays.')
        self.tune_internal = self._build_automodel(data, validation_data, batch_size, epochs)
        self.trainer = Trainer(logger=False, max_epochs=epochs, checkpoint_callback=self.checkpoint_callback, num_processes=self.num_processes, use_ipex=self.use_ipex, use_hpo=True)
        self.internal = self.trainer.search(self.tune_internal, n_trials=n_trials, target_metric=formated_target_metric, direction=direction, directions=directions, n_parallels=n_parallels, acceleration=acceleration, input_sample=input_sample, **kwargs)
        if self.trainer.hposearcher.objective.mo_hpo:
            return self.internal

    def search_summary(self):
        if False:
            i = 10
            return i + 15
        '\n        Return search summary of HPO.\n        '
        invalidOperationError(self.use_hpo, 'No search summary when HPO is disabled.')
        return self.trainer.search_summary()

    def fit(self, data, validation_data=None, epochs=1, batch_size=32, validation_mode='output', earlystop_patience=1, use_trial_id=None):
        if False:
            print('Hello World!')
        "\n        Fit(Train) the forecaster.\n\n        :param data: The data support following formats:\n\n               | 1. numpy ndarrays: generate from `TSDataset.roll`,\n                    be sure to set label_len > 0 and time_enc = True\n               | 2. pytorch dataloader: generate from `TSDataset.to_torch_data_loader`,\n                    be sure to set label_len > 0 and time_enc = True\n               | 3. A bigdl.chronos.data.tsdataset.TSDataset instance\n\n        :param validation_data: Validation sample for validation loop. Defaults to 'None'.\n               If you do not input data for 'validation_data', the validation_step will be skipped.\n               The validation_data support following formats:\n\n               | 1. numpy ndarrays: generate from `TSDataset.roll`,\n                    be sure to set label_len > 0 and time_enc = True\n               | 2. pytorch dataloader: generate from `TSDataset.to_torch_data_loader`,\n                    be sure to set label_len > 0 and time_enc = True\n               | 3. A bigdl.chronos.data.tsdataset.TSDataset instance\n\n        :param epochs: Number of epochs you want to train. The value defaults to 1.\n        :param batch_size: Number of batch size you want to train. The value defaults to 32.\n               if you input a pytorch dataloader for `data`, the batch_size will follow the\n               batch_size setted in `data`.\n        :param validation_mode:  A str represent the operation mode while having 'validation_data'.\n               Defaults to 'output'. The validation_mode includes the following types:\n\n               | 1. output:\n               | If you choose 'output' for validation_mode, it will return a dict that records the\n               | average validation loss of each epoch.\n               |\n               | 2. earlystop:\n               | Monitor the val_loss and stop training when it stops improving.\n               |\n               | 3. best_epoch:\n               | Monitor the val_loss. And load the checkpoint of the epoch with the smallest\n               | val_loss after the training.\n\n        :param earlystop_patience: Number of checks with no improvement after which training will\n               be stopped. It takes effect when 'validation_mode' is 'earlystop'. Under the default\n               configuration, one check happens after every training epoch.\n        :param use_trail_id: choose a internal according to trial_id, which is used only\n               in multi-objective search.\n        "
        if self.distributed:
            invalidInputError(False, 'distributed is not support in Autoformer')
        if isinstance(data, tuple):
            data = DataLoader(TensorDataset(torch.from_numpy(data[0]), torch.from_numpy(data[1]), torch.from_numpy(data[2]), torch.from_numpy(data[3])), batch_size=batch_size, shuffle=True)
        if isinstance(data, TSDataset):
            _rolled = data.numpy_x is None
            data = data.to_torch_data_loader(batch_size=batch_size, roll=_rolled, lookback=self.data_config['past_seq_len'], horizon=self.data_config['future_seq_len'], label_len=self.data_config['label_len'], time_enc=True, feature_col=data.roll_feature, target_col=data.roll_target, shuffle=True)
        from bigdl.chronos.pytorch import TSTrainer as Trainer
        if self.use_hpo is True:
            invalidOperationError(hasattr(self, 'trainer'), 'There is no trainer, and you should call .tune() before .fit()')
            if self.trainer.hposearcher.objective.mo_hpo:
                invalidOperationError(self.trainer.hposearcher.study, 'You must tune before fit the model.')
                invalidInputError(use_trial_id is not None, 'For multibojective HPO, you must specify a trial id for fit.')
                trial = self.trainer.hposearcher.study.trials[use_trial_id]
                self.internal = self.tune_internal._model_build(trial)
        with TemporaryDirectory() as forecaster_log_dir:
            with TemporaryDirectory() as validation_ckpt_dir:
                from pytorch_lightning.loggers import CSVLogger
                logger = False if validation_data is None else CSVLogger(save_dir=forecaster_log_dir, flush_logs_every_n_steps=10, name='forecaster_tmp_log')
                from pytorch_lightning.callbacks import EarlyStopping
                early_stopping = EarlyStopping('val_loss', patience=earlystop_patience)
                from pytorch_lightning.callbacks import ModelCheckpoint
                checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath=validation_ckpt_dir, filename='best', save_on_train_epoch_end=True)
                if validation_mode == 'earlystop':
                    callbacks = [early_stopping]
                elif validation_mode == 'best_epoch':
                    callbacks = [checkpoint_callback]
                else:
                    callbacks = None
                self.trainer = Trainer(logger=logger, max_epochs=epochs, callbacks=callbacks, enable_checkpointing=self.checkpoint_callback, num_processes=self.num_processes, use_ipex=self.use_ipex, log_every_n_steps=10)
                if validation_data is None:
                    self.trainer.fit(self.internal, data)
                    self.fitted = True
                else:
                    if isinstance(validation_data, tuple):
                        validation_data = DataLoader(TensorDataset(torch.from_numpy(validation_data[0]), torch.from_numpy(validation_data[1]), torch.from_numpy(validation_data[2]), torch.from_numpy(validation_data[3])), batch_size=batch_size, shuffle=False)
                    if isinstance(validation_data, TSDataset):
                        _rolled = validation_data.numpy_x is None
                        validation_data = validation_data.to_torch_data_loader(batch_size=batch_size, roll=_rolled, lookback=self.data_config['past_seq_len'], horizon=self.data_config['future_seq_len'], label_len=self.data_config['label_len'], time_enc=True, feature_col=validation_data.roll_feature, target_col=validation_data.roll_target, shuffle=False)
                    self.trainer.fit(self.internal, data, validation_data)
                    self.fitted = True
                    fit_csv = os.path.join(forecaster_log_dir, 'forecaster_tmp_log/version_0/metrics.csv')
                    best_path = os.path.join(validation_ckpt_dir, 'best.ckpt')
                    fit_out = read_csv(fit_csv, loss_name='val_loss')
                    if validation_mode == 'best_epoch':
                        self.load(best_path)
                    self.trainer._logger_connector.on_trainer_init(False, self.trainer.flush_logs_every_n_steps, self.trainer.log_every_n_steps, self.trainer.move_metrics_to_cpu)
                    return fit_out

    def get_context(self, thread_num=None):
        if False:
            while True:
                i = 10
        '\n        Obtain context manager from forecaster.\n        :param thread_num: int, the num of thread limit. The value is set to None by\n               default where no limit is set.\n        :return: a context manager.\n        '
        return ForecasterContextManager(self, thread_num, optimize=False)

    def predict(self, data, batch_size=32):
        if False:
            while True:
                i = 10
        '\n        Predict using a trained forecaster.\n\n        :param data: The data support following formats:\n\n               | 1. numpy ndarrays: generate from `TSDataset.roll`,\n                    be sure to set label_len > 0 and time_enc = True\n               | 2. pytorch dataloader: generate from `TSDataset.to_torch_data_loader`,\n                    be sure to set label_len > 0, time_enc = True and is_predict = True\n               | 3. A bigdl.chronos.data.tsdataset.TSDataset instance\n\n        :param batch_size: predict batch size. The value will not affect predict\n               result but will affect resources cost(e.g. memory and time).\n\n        :return: A list of numpy ndarray\n        '
        if self.distributed:
            invalidInputError(False, 'distributed is not support in Autoformer')
        if isinstance(data, TSDataset):
            _rolled = data.numpy_x is None
            data = data.to_torch_data_loader(batch_size=batch_size, roll=_rolled, lookback=self.data_config['past_seq_len'], horizon=self.data_config['future_seq_len'], label_len=self.data_config['label_len'], time_enc=True, feature_col=data.roll_feature, target_col=data.roll_target, shuffle=False)
        invalidInputError(isinstance(data, tuple) or isinstance(data, DataLoader), f'The input data to predict() support formats: numpy ndarray tuple and pytorch dataloader, but found {type(data)}.')
        if isinstance(data, tuple):
            data = DataLoader(TensorDataset(torch.from_numpy(data[0]), torch.from_numpy(data[1]), torch.from_numpy(data[2]), torch.from_numpy(data[3])), batch_size=batch_size, shuffle=False)
        if not self.context_enabled:
            self.cxt_manager = ForecasterContextManager(self, self.thread_num, optimize=False)
        with self.cxt_manager:
            return self.trainer.predict(self.internal, data)

    def evaluate(self, data, batch_size=32):
        if False:
            while True:
                i = 10
        '\n        Predict using a trained forecaster.\n\n        :param data: The data support following formats:\n\n               | 1. numpy ndarrays: generate from `TSDataset.roll`,\n                    be sure to set label_len > 0 and time_enc = True\n               | 2. pytorch dataloader: generate from `TSDataset.to_torch_data_loader`,\n                    be sure to set label_len > 0 and time_enc = True\n               | 3. A bigdl.chronos.data.tsdataset.TSDataset instance\n\n        :param batch_size: predict batch size. The value will not affect predict\n               result but will affect resources cost(e.g. memory and time).\n\n        :return: A dict, currently returns the loss rather than metrics\n        '
        if self.distributed:
            invalidInputError(False, 'distributed is not support in Autoformer')
        if isinstance(data, TSDataset):
            _rolled = data.numpy_x is None
            data = data.to_torch_data_loader(batch_size=batch_size, roll=_rolled, lookback=self.data_config['past_seq_len'], horizon=self.data_config['future_seq_len'], label_len=self.data_config['label_len'], time_enc=True, feature_col=data.roll_feature, target_col=data.roll_target, shuffle=False)
        invalidInputError(isinstance(data, tuple) or isinstance(data, DataLoader), f'The input data to predict() support formats: numpy ndarray tuple and pytorch dataloader, but found {type(data)}.')
        if isinstance(data, tuple):
            data = DataLoader(TensorDataset(torch.from_numpy(data[0]), torch.from_numpy(data[1]), torch.from_numpy(data[2]), torch.from_numpy(data[3])), batch_size=batch_size, shuffle=False)
        return self.trainer.validate(self.internal, data)

    def predict_interval(self, data, validation_data=None, batch_size=32, repetition_times=5):
        if False:
            return 10
        '\n        Calculate confidence interval of data based on Monte Carlo dropout(MC dropout).\n        Related paper : https://arxiv.org/abs/1709.01907\n\n        :param data: The data support following formats:\n\n               | 1. numpy ndarrays: generate from `TSDataset.roll`,\n                    be sure to set label_len > 0 and time_enc = True\n               | 2. pytorch dataloader: generate from `TSDataset.to_torch_data_loader`,\n                    be sure to set label_len > 0, time_enc = True\n               | 3. A bigdl.chronos.data.tsdataset.TSDataset instance\n\n        :param validation_data: The validation_data support following formats:\n\n               | 1. numpy ndarrays: generate from `TSDataset.roll`,\n                    be sure to set label_len > 0 and time_enc = True\n               | 2. pytorch dataloader: generate from `TSDataset.to_torch_data_loader`,\n                    be sure to set label_len > 0, time_enc = True\n               | 3. A bigdl.chronos.data.tsdataset.TSDataset instance\n\n        :param batch_size: predict batch size. The value will not affect predict\n               result but will affect resources cost(e.g. memory and time).\n        :param repetition_times : Defines repeate how many times to calculate model\n                                  uncertainty based on MC Dropout.\n\n        :return: prediction and standard deviation which are both numpy array\n                 with shape (num_samples, horizon, target_dim)\n\n        '
        from bigdl.chronos.pytorch.utils import _pytorch_fashion_inference
        if self.fitted is not True:
            invalidInputError(False, 'You must call fit or restore first before calling predict_interval!')
        if not hasattr(self, 'data_noise'):
            invalidInputError(validation_data is not None, 'When call predict_interval for the first time, you must pass in validation_data to calculate data noise.')
            if isinstance(validation_data, TSDataset):
                _rolled = validation_data.numpy_x is None
                validation_data = validation_data.to_torch_data_loader(batch_size=batch_size, roll=_rolled, lookback=self.data_config['past_seq_len'], horizon=self.data_config['future_seq_len'], label_len=self.data_config['label_len'], time_enc=True, feature_col=data.roll_feature, target_col=data.roll_target, shuffle=False)
            if isinstance(validation_data, DataLoader):
                target = np.concatenate(tuple((val[1] for val in validation_data)), axis=0)
            else:
                (_, target, _, _) = validation_data
            target = target[:, -self.data_config['future_seq_len']:, :]
            _yhat = self.predict(validation_data)
            val_yhat = np.concatenate(_yhat, axis=0)
            self.data_noise = Evaluator.evaluate(['mse'], target, val_yhat, aggregate=None)[0]

        def apply_dropout(m):
            if False:
                for i in range(10):
                    print('nop')
            if type(m) == torch.nn.Dropout:
                m.train()
        self.internal.apply(apply_dropout)
        if isinstance(data, TSDataset):
            _rolled = data.numpy_x is None
            data = data.to_torch_data_loader(batch_size=batch_size, roll=_rolled, lookback=self.data_config['past_seq_len'], horizon=self.data_config['future_seq_len'], label_len=self.data_config['label_len'], time_enc=True, feature_col=data.roll_feature, target_col=data.roll_target, shuffle=False)

        def predict(data, model):
            if False:
                while True:
                    i = 10
            if isinstance(data, tuple):
                data = DataLoader(TensorDataset(torch.from_numpy(data[0]), torch.from_numpy(data[1]), torch.from_numpy(data[2]), torch.from_numpy(data[3])), batch_size=batch_size, shuffle=False)
            outputs_list = []
            for batch in data:
                (batch_x, batch_y, batch_x_mark, batch_y_mark) = map(lambda x: x.float(), batch)
                outputs = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                outputs = outputs[:, -model.pred_len:, -model.c_out:]
                outputs_list.append(outputs.detach().numpy())
            return outputs_list
        y_hat_list = []
        for i in range(repetition_times):
            _yhat = predict(data, self.internal)
            yhat = np.concatenate(_yhat, axis=0)
            y_hat_list.append(yhat)
        y_hat_mean = np.mean(np.stack(y_hat_list, axis=0), axis=0)
        model_bias = np.zeros_like(y_hat_mean)
        for i in range(repetition_times):
            model_bias += (y_hat_list[i] - y_hat_mean) ** 2
        model_bias /= repetition_times
        std_deviation = np.sqrt(self.data_noise + model_bias)
        return (y_hat_mean, std_deviation)

    def get_model(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the learned PyTorch Lightning model.\n\n        :return: a pytorch lightning model instance\n        '
        return self.internal

    def load(self, checkpoint_file):
        if False:
            return 10
        '\n        restore the forecaster.\n\n        :param checkpoint_file: The checkpoint file location you want to load the forecaster.\n        '
        self.trainer = Trainer(logger=False, max_epochs=1, checkpoint_callback=self.checkpoint_callback, num_processes=1, use_ipex=self.use_ipex, distributed_backend='spawn')
        checkpoint = torch.load(checkpoint_file)
        config = checkpoint['hyper_parameters']
        args = _transform_config_to_namedtuple(config)
        internal = AutoFormer.load_from_checkpoint(checkpoint_file, configs=args)
        self.internal = internal

    def save(self, checkpoint_file):
        if False:
            while True:
                i = 10
        '\n        save the forecaster.\n\n        :param checkpoint_file: The checkpoint file location you want to load the forecaster.\n        '
        if self.use_hpo:
            self.trainer.model = self.trainer.model.model
        self.trainer.save_checkpoint(checkpoint_file)

    @classmethod
    def from_tsdataset(cls, tsdataset, past_seq_len=None, future_seq_len=None, label_len=None, freq=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Build a Forecaster Model.\n\n        :param tsdataset: A bigdl.chronos.data.tsdataset.TSDataset instance.\n        :param past_seq_len: int or "auto", Specify the history time steps (i.e. lookback).\n               Do not specify the \'past_seq_len\' if your tsdataset has called\n               the \'TSDataset.roll\' method or \'TSDataset.to_torch_data_loader\'.\n               If "auto", the mode of time series\' cycle length will be taken as the past_seq_len.\n        :param future_seq_len: int or list, Specify the output time steps (i.e. horizon).\n               Do not specify the \'future_seq_len\' if your tsdataset has called\n               the \'TSDataset.roll\' method or \'TSDataset.to_torch_data_loader\'.\n        :param kwargs: Specify parameters of Forecaster,\n               e.g. loss and optimizer, etc.\n               More info, please refer to Forecaster.__init__ methods.\n\n        :return: A Forecaster Model.\n        '
        from bigdl.nano.utils.common import invalidInputError
        invalidInputError(isinstance(tsdataset, TSDataset), f'We only supports input a TSDataset, but get{type(tsdataset)}.')

        def check_time_steps(tsdataset, past_seq_len, future_seq_len):
            if False:
                while True:
                    i = 10
            if tsdataset.lookback is not None and past_seq_len is not None:
                future_seq_len = future_seq_len if isinstance(future_seq_len, int) else max(future_seq_len)
                return tsdataset.lookback == past_seq_len and tsdataset.horizon == future_seq_len
            return True
        invalidInputError(not tsdataset._has_generate_agg_feature, "We will add support for 'gen_rolling_feature' method later.")
        if tsdataset.lookback is not None:
            past_seq_len = tsdataset.lookback
            future_seq_len = tsdataset.horizon if isinstance(tsdataset.horizon, int) else max(tsdataset.horizon)
            output_feature_num = len(tsdataset.roll_target)
            input_feature_num = len(tsdataset.roll_feature) + output_feature_num
        elif past_seq_len is not None and future_seq_len is not None:
            past_seq_len = past_seq_len if isinstance(past_seq_len, int) else tsdataset.get_cycle_length()
            future_seq_len = future_seq_len if isinstance(future_seq_len, int) else max(future_seq_len)
            output_feature_num = len(tsdataset.target_col)
            input_feature_num = len(tsdataset.feature_col) + output_feature_num
        else:
            invalidInputError(False, "Forecaster requires 'past_seq_len' and 'future_seq_len' to specify the history time step and output time step.")
        if label_len is None:
            label_len = max(past_seq_len // 2, 1)
        invalidInputError(tsdataset.label_len == label_len or tsdataset.label_len is None, f'Expected label_len to be {tsdataset.label_len}, but found {label_len}')
        invalidInputError(check_time_steps(tsdataset, past_seq_len, future_seq_len), f'tsdataset already has history time steps and differs from the given past_seq_len and future_seq_len Expected past_seq_len and future_seq_len to be {(tsdataset.lookback, tsdataset.horizon)}, but found {(past_seq_len, future_seq_len)}.', fixMsg='Do not specify past_seq_len and future seq_len or call tsdataset.roll method again and specify time step')
        if tsdataset._freq is not None:
            infer_freq_str = _timedelta_to_delta_str(tsdataset._freq)
            freq = infer_freq_str
        return cls(past_seq_len=past_seq_len, future_seq_len=future_seq_len, input_feature_num=input_feature_num, output_feature_num=output_feature_num, freq=freq, label_len=label_len, **kwargs)

def _str2metric(metric):
    if False:
        while True:
            i = 10
    if isinstance(metric, str):
        metric_name = metric
        from bigdl.chronos.metric.forecast_metrics import REGRESSION_MAP
        metric_func = REGRESSION_MAP[metric_name]

        def metric(y_label, y_predict):
            if False:
                i = 10
                return i + 15
            y_label = y_label.numpy()
            y_predict = y_predict.numpy()
            return metric_func(y_label, y_predict)
        metric.__name__ = metric_name
    return metric

def _timedelta_to_delta_str(offset):
    if False:
        return 10
    features_by_offsets = ((Timedelta(seconds=60), 's'), (Timedelta(minutes=60), 't'), (Timedelta(hours=24), 'h'), (Timedelta(days=7), 'd'), (Timedelta(days=30), 'w'), (Timedelta(days=365), 'm'))
    for (offset_type, offset_str) in features_by_offsets:
        if offset < offset_type:
            return offset_str
    return 'a'