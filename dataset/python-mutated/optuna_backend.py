from bigdl.nano.automl.hpo.backend import PrunerType, SamplerType
from bigdl.nano.utils.common import invalidInputError
from bigdl.nano.automl.hpo.space import SimpleSpace, NestedSpace, AutoObject
from bigdl.nano.automl.hpo.space import AutoObject, Space, SingleParam, _get_hp_prefix
import optuna

class OptunaBackend(object):
    """A Wrapper to shield user from Optuna specific configurations and API      Later may support other HPO search engines."""
    pruner_map = {PrunerType.HyperBand: optuna.pruners.HyperbandPruner, PrunerType.Median: optuna.pruners.MedianPruner, PrunerType.Nop: optuna.pruners.NopPruner, PrunerType.Patient: optuna.pruners.PatientPruner, PrunerType.Percentile: optuna.pruners.PercentilePruner, PrunerType.SuccessiveHalving: optuna.pruners.SuccessiveHalvingPruner, PrunerType.Threshold: optuna.pruners.ThresholdPruner}
    sampler_map = {SamplerType.TPE: optuna.samplers.TPESampler, SamplerType.CmaEs: optuna.samplers.CmaEsSampler, SamplerType.Grid: optuna.samplers.GridSampler, SamplerType.Random: optuna.samplers.RandomSampler, SamplerType.PartialFixed: optuna.samplers.PartialFixedSampler, SamplerType.NSGAII: optuna.samplers.NSGAIISampler, SamplerType.MOTPE: optuna.samplers.MOTPESampler}
    SPLITTER = u':'

    @staticmethod
    def get_other_args(kwargs, kwspaces):
        if False:
            return 10
        'Get key-word arguments which are not search spaces.'
        return {k: kwargs[k] for k in set(kwargs) - set(kwspaces)}

    @staticmethod
    def _sample_space(trial, hp_name, hp_obj):
        if False:
            print('Hello World!')
        hp_type = str(type(hp_obj)).lower()
        if 'integer' in hp_type or 'float' in hp_type or 'categorical' in hp_type or ('ordinal' in hp_type):
            try:
                if 'integer' in hp_type:
                    hp_dimension = trial.suggest_int(name=hp_name, low=int(hp_obj.lower), high=int(hp_obj.upper))
                elif 'float' in hp_type:
                    if hp_obj.log:
                        hp_dimension = trial.suggest_loguniform(name=hp_name, low=float(hp_obj.lower), high=float(hp_obj.upper))
                    else:
                        hp_dimension = trial.suggest_float(name=hp_name, low=float(hp_obj.lower), high=float(hp_obj.upper))
                elif 'categorical' in hp_type:
                    hp_dimension = trial.suggest_categorical(name=hp_name, choices=hp_obj.choices)
                elif 'ordinal' in hp_type:
                    hp_dimension = trial.suggest_categorical(name=hp_name, choices=hp_obj.sequence)
            except RuntimeError:
                invalidInputError(False, 'If you set search space in model, you must call model.search before model.fit.')
        else:
            invalidInputError(False, 'unknown hyperparameter type %s for param %s' % (hp_type, hp_name))
        return hp_dimension

    @staticmethod
    def get_hpo_config(trial, configspace):
        if False:
            i = 10
            return i + 15
        'Get hyper parameter suggestions from search space settings.'
        hp_ordering = configspace.get_hyperparameter_names()
        config = {}
        for hp_name in hp_ordering:
            hp = configspace.get_hyperparameter(hp_name)
            hp_prefix = _get_hp_prefix(hp)
            optuna_hp_name = OptunaBackend._format_hp_name(hp_prefix, hp_name)
            hp_dimension = OptunaBackend._sample_space(trial, optuna_hp_name, hp)
            config[hp_name] = hp_dimension
        return config

    @staticmethod
    def _format_hp_name(prefix, hp_name):
        if False:
            i = 10
            return i + 15
        if prefix:
            return '{}{}{}'.format(prefix, OptunaBackend.SPLITTER, hp_name)
        else:
            return hp_name

    @staticmethod
    def instantiate_param(trial, kwargs, arg_name):
        if False:
            while True:
                i = 10
        '\n        Instantiate auto objects in kwargs with trial params at runtime.\n\n        Note the params are replaced IN-PLACE\n        '
        v = kwargs.get(arg_name, None)
        if not v:
            return kwargs
        if not isinstance(v, Space):
            value = v
        elif isinstance(v, AutoObject):
            value = OptunaBackend.instantiate(trial, v)
        else:
            pobj = SingleParam(arg_name, v)
            config = OptunaBackend.get_hpo_config(trial, pobj.cs)
            value = pobj.sample(**config)
        kwargs[arg_name] = value
        return kwargs

    @staticmethod
    def instantiate(trial, lazyobj):
        if False:
            for i in range(10):
                print('nop')
        "Instantiate a lazyobject from a trial's sampled param set."
        config = OptunaBackend.gen_config(trial, lazyobj)
        return lazyobj.sample(**config)

    @staticmethod
    def gen_config(trial, automl_obj):
        if False:
            while True:
                i = 10
        "Generate the param config from a trial's sampled param set."
        configspace = automl_obj.cs
        config = OptunaBackend.get_hpo_config(trial, configspace)
        other_kwargs = OptunaBackend.get_other_args(automl_obj.kwargs, automl_obj.kwspaces)
        config.update(other_kwargs)
        return config

    @staticmethod
    def create_sampler(sampler_type, kwargs):
        if False:
            while True:
                i = 10
        'Create a hyperparameter sampler by type.'
        sampler_class = OptunaBackend.sampler_map.get(sampler_type)
        return sampler_class(kwargs)

    @staticmethod
    def create_pruner(pruner_type, kwargs):
        if False:
            while True:
                i = 10
        'Create a pruner by type.'
        pruner_class = OptunaBackend.pruner_map.get(pruner_type)
        return pruner_class(**kwargs)

    @staticmethod
    def create_study(**kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Create a study to drive the hyperparameter search.'
        return optuna.create_study(**kwargs)