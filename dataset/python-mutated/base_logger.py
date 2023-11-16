from abc import ABC
from copy import deepcopy
from sklearn.pipeline import Pipeline
SETUP_TAG = 'Session Initialized'

class BaseLogger(ABC):

    def init_logger(self):
        if False:
            print('Hello World!')
        pass

    def __del__(self):
        if False:
            i = 10
            return i + 15
        try:
            self.finish_experiment()
        except Exception:
            pass

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return self.__class__.__name__

    def log_params(self, params, model_name=None):
        if False:
            for i in range(10):
                print('nop')
        pass

    def init_experiment(self, exp_name_log, full_name=None, setup=True, **kwargs):
        if False:
            return 10
        pass

    def set_tags(self, source, experiment_custom_tags, runtime, USI=None):
        if False:
            return 10
        pass

    def _construct_pipeline_if_needed(self, model, prep_pipe: Pipeline) -> Pipeline:
        if False:
            i = 10
            return i + 15
        'If model is a pipeline, return it, else append model to copy of prep_pipe.'
        if not isinstance(model, Pipeline):
            prep_pipe_temp = deepcopy(prep_pipe)
            prep_pipe_temp.steps.append(['trained_model', model])
        else:
            prep_pipe_temp = model
        return prep_pipe_temp

    def log_sklearn_pipeline(self, experiment, prep_pipe, model, path=None):
        if False:
            print('Hello World!')
        pass

    def log_model_comparison(self, model_result, source):
        if False:
            while True:
                i = 10
        pass

    def log_metrics(self, metrics, source=None):
        if False:
            for i in range(10):
                print('nop')
        pass

    def log_plot(self, plot, title):
        if False:
            return 10
        pass

    def log_hpram_grid(self, html_file, title='hpram_grid'):
        if False:
            for i in range(10):
                print('nop')
        pass

    def log_artifact(self, file, type='artifact'):
        if False:
            for i in range(10):
                print('nop')
        pass

    def finish_experiment(self):
        if False:
            i = 10
            return i + 15
        pass