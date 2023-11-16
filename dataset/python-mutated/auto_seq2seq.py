from .base_automodel import BaseAutomodel

class AutoSeq2Seq(BaseAutomodel):

    def __init__(self, input_feature_num, output_target_num, past_seq_len, future_seq_len, optimizer, loss, metric, metric_mode=None, lr=0.001, lstm_hidden_dim=128, lstm_layer_num=2, dropout=0.25, teacher_forcing=False, backend='torch', logs_dir='/tmp/auto_seq2seq', cpus_per_trial=1, name='auto_seq2seq', remote_dir=None):
        if False:
            return 10
        '\n        Create an AutoSeq2Seq.\n\n        :param input_feature_num: Int. The number of features in the input\n        :param output_target_num: Int. The number of targets in the output\n        :param past_seq_len: Int. The number of historical steps used for forecasting.\n        :param future_seq_len: Int. The number of future steps to forecast.\n        :param optimizer: String or pyTorch optimizer creator function or\n               tf.keras optimizer instance.\n        :param loss: String or pytorch/tf.keras loss instance or pytorch loss creator function.\n        :param metric: String or customized evaluation metric function.\n               If string, metric is the evaluation metric name to optimize, e.g. "mse".\n               If callable function, it signature should be func(y_true, y_pred), where y_true and\n               y_pred are numpy ndarray. The function should return a float value\n               as evaluation result.\n        :param metric_mode: One of ["min", "max"]. "max" means greater metric value is better.\n               You have to specify metric_mode if you use a customized metric function.\n               You don\'t have to specify metric_mode if you use the built-in metric in\n               bigdl.orca.automl.metrics.Evaluator.\n        :param lr: float or hp sampling function from a float space. Learning rate.\n               e.g. hp.choice([0.001, 0.003, 0.01])\n        :param lstm_hidden_dim: LSTM hidden channel for decoder and encoder.\n               hp.grid_search([32, 64, 128])\n        :param lstm_layer_num: LSTM layer number for decoder and encoder.\n               e.g. hp.randint(1, 4)\n        :param dropout: float or hp sampling function from a float space. Learning rate. Dropout\n               rate. e.g. hp.uniform(0.1, 0.3)\n        :param teacher_forcing: If use teacher forcing in training. e.g. hp.choice([True, False])\n        :param backend: The backend of the Seq2Seq model. support "keras" and "torch".\n        :param logs_dir: Local directory to save logs and results. It defaults to\n               "/tmp/auto_seq2seq"\n        :param cpus_per_trial: Int. Number of cpus for each trial. It defaults to 1.\n        :param name: name of the AutoSeq2Seq. It defaults to "auto_seq2seq"\n        :param remote_dir: String. Remote directory to sync training results and checkpoints. It\n               defaults to None and doesn\'t take effects while running in local. While running in\n               cluster, it defaults to "hdfs:///tmp/{name}".\n        '
        self.search_space = dict(input_feature_num=input_feature_num, output_feature_num=output_target_num, past_seq_len=past_seq_len, future_seq_len=future_seq_len, lstm_hidden_dim=lstm_hidden_dim, lstm_layer_num=lstm_layer_num, lr=lr, dropout=dropout, teacher_forcing=teacher_forcing)
        self.metric = metric
        self.metric_mode = metric_mode
        self.backend = backend
        self.optimizer = optimizer
        self.loss = loss
        self._auto_est_config = dict(logs_dir=logs_dir, resources_per_trial={'cpu': cpus_per_trial}, remote_dir=remote_dir, name=name)
        if self.backend.startswith('torch'):
            from bigdl.chronos.model.Seq2Seq_pytorch import model_creator
        elif self.backend.startswith('keras'):
            from bigdl.chronos.model.tf2.Seq2Seq_keras import model_creator_auto as model_creator
        else:
            from bigdl.nano.utils.common import invalidInputError
            invalidInputError(False, f'We only support keras and torch as backend, but got {self.backend}')
        self._model_creator = model_creator
        super().__init__()