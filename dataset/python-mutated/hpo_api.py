import warnings

def create_hpo_searcher(trainer, num_processes=1):
    if False:
        while True:
            i = 10
    'Create HPO Search for PyTorch.'
    from bigdl.nano.automl.pytorch import HPOSearcher
    return HPOSearcher(trainer, num_processes=num_processes)

def check_hpo_status(searcher):
    if False:
        return 10
    'Check the status of hpo.'
    if not searcher:
        warnings.warn('HPO is not properly enabled or required                 dependency is not installed.', UserWarning)
        return False
    return True

def create_optuna_backend():
    if False:
        print('Hello World!')
    'Create an Optuna Backend.'
    from bigdl.nano.deps.automl.optuna_backend import OptunaBackend
    return OptunaBackend()

def create_optuna_pl_pruning_callback(*args, **kwargs):
    if False:
        i = 10
        return i + 15
    'Create PyTorchLightning Pruning Callback.'
    from optuna.integration import PyTorchLightningPruningCallback
    return PyTorchLightningPruningCallback(*args, **kwargs)

def create_optuna_tfkeras_pruning_callback(*args, **kwargs):
    if False:
        return 10
    'Create Tensorflow Keras Pruning Callback.'
    from optuna.integration import TFKerasPruningCallback
    return TFKerasPruningCallback(*args, **kwargs)

def create_configuration_space(*args, **kwargs):
    if False:
        print('Hello World!')
    'Create Configuration Space.'
    import ConfigSpace as CS
    return CS.ConfigurationSpace(*args, **kwargs)

def create_categorical_hp(*args, **kwargs):
    if False:
        i = 10
        return i + 15
    'Create Categorical Hyperparamter.'
    import ConfigSpace.hyperparameters as CSH
    return CSH.CategoricalHyperparameter(*args, **kwargs)

def create_uniform_float_hp(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Create UniformFloat Hyperparameter.'
    import ConfigSpace.hyperparameters as CSH
    return CSH.UniformFloatHyperparameter(*args, **kwargs)

def create_uniform_int_hp(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Create UniformFloat Hyperparameter.'
    import ConfigSpace.hyperparameters as CSH
    return CSH.UniformIntegerHyperparameter(*args, **kwargs)