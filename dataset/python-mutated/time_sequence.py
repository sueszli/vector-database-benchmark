import os
import time
from bigdl.chronos.autots.deprecated.feature.utils import save_config
from bigdl.chronos.autots.deprecated.pipeline.base import Pipeline
from bigdl.chronos.autots.deprecated.model.time_sequence import TimeSequenceModel
from bigdl.chronos.autots.deprecated.pipeline.parameters import DEFAULT_CONFIG_DIR, DEFAULT_PPL_DIR
from bigdl.chronos.utils import deprecated

class TimeSequencePipeline(Pipeline):

    def __init__(self, model=None, name=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        initialize a pipeline\n        :param model: the internal model\n        '
        self.model = model
        self.config = self.model.config
        self.name = name
        self.time = time.strftime('%Y%m%d-%H%M%S')

    def describe(self):
        if False:
            for i in range(10):
                print('nop')
        init_info = ['future_seq_len', 'dt_col', 'target_col', 'extra_features_col', 'drop_missing']
        print('**** Initialization info ****')
        for info in init_info:
            print(info + ':', getattr(self.model.ft, info))
        print('')

    def fit(self, input_df, validation_df=None, mc=False, epoch_num=20):
        if False:
            return 10
        self.model.fit_incr(input_df, validation_df, mc=mc, verbose=1, epochs=epoch_num)
        print('Fit done!')

    def fit_with_fixed_configs(self, input_df, validation_df=None, mc=False, **user_configs):
        if False:
            i = 10
            return i + 15
        '\n        Fit pipeline with fixed configs. The model will be trained from initialization\n        with the hyper-parameter specified in configs. The configs contain both identity configs\n        (Eg. "future_seq_len", "dt_col", "target_col", "metric") and automl tunable configs\n        (Eg. "past_seq_len", "batch_size").\n        We recommend calling get_default_configs to see the name and default values of configs you\n        you can specify.\n        :param input_df: one data frame or a list of data frames\n        :param validation_df: one data frame or a list of data frames\n        :param user_configs: you can overwrite or add more configs with user_configs. Eg. "epochs"\n        :return:\n        '
        config = self.config.copy()
        config.update(user_configs)
        self.model.setup(config)
        self.model.fit_eval(data=input_df, validation_data=validation_df, mc=mc, verbose=1, **config)

    def evaluate(self, input_df, metrics=['mse'], multioutput='raw_values'):
        if False:
            for i in range(10):
                print('nop')
        "\n        evaluate the pipeline\n        :param input_df:\n        :param metrics: subset of ['mean_squared_error', 'r_square', 'sMAPE']\n        :param multioutput: string in ['raw_values', 'uniform_average']\n                'raw_values' :\n                    Returns a full set of errors in case of multioutput input.\n                'uniform_average' :\n                    Errors of all outputs are averaged with uniform weight.\n        :return:\n        "
        return self.model.evaluate(df=input_df, metric=metrics)

    def predict(self, input_df):
        if False:
            for i in range(10):
                print('nop')
        '\n        predict test data with the pipeline fitted\n        :param input_df:\n        :return:\n        '
        return self.model.predict(df=input_df)

    def predict_with_uncertainty(self, input_df, n_iter=100):
        if False:
            i = 10
            return i + 15
        return self.model.predict_with_uncertainty(input_df, n_iter)

    def save(self, ppl_file=None):
        if False:
            return 10
        '\n        save pipeline to file, contains feature transformer, model, trial config.\n        :param ppl_file:\n        :return:\n        '
        ppl_file = ppl_file or os.path.join(DEFAULT_PPL_DIR, '{}_{}.ppl'.format(self.name, self.time))
        self.model.save(ppl_file)
        return ppl_file

    def config_save(self, config_file=None):
        if False:
            while True:
                i = 10
        '\n        save all configs to file.\n        :param config_file:\n        :return:\n        '
        config_file = config_file or os.path.join(DEFAULT_CONFIG_DIR, '{}_{}.json'.format(self.name, self.time))
        save_config(config_file, self.config, replace=True)
        return config_file

@deprecated('Please use `bigdl.chronos.autots.TSPipeline` instead.')
def load_ts_pipeline(file):
    if False:
        for i in range(10):
            print('nop')
    model = TimeSequenceModel()
    model.restore(file)
    print('Restore pipeline from', file)
    return TimeSequencePipeline(model)