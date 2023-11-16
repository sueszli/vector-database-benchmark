import logging
from torch.ao.pruning._experimental.data_sparsifier.base_data_sparsifier import SUPPORTED_TYPES
logger: logging.Logger = logging.getLogger(__name__)

def _attach_model_to_data_sparsifier(module, data_sparsifier, config=None):
    if False:
        print('Hello World!')
    "Attaches a data sparsifier to all the layers of the module.\n    Essentially, loop over all the weight parameters in the module and\n    attach it to the data sparsifier.\n    Note::\n        The '.' in the layer names are replaced with '_' (refer to _get_valid_name() below)\n        before attaching to the sparsifier. This is because, the data\n        sparsifier uses a dummy model inside to store the weight parameters.\n    "
    if config is None:
        config = {}
    for (name, parameter) in module.named_parameters():
        if type(parameter) in SUPPORTED_TYPES:
            valid_name = _get_valid_name(name)
            data_sparsifier.add_data(name=valid_name, data=parameter, **config.get(valid_name, {}))

def _get_valid_name(name):
    if False:
        return 10
    return name.replace('.', '_')

def _log_sparsified_level(model, data_sparsifier) -> None:
    if False:
        while True:
            i = 10
    for (name, parameter) in model.named_parameters():
        if type(parameter) not in SUPPORTED_TYPES:
            continue
        valid_name = _get_valid_name(name)
        mask = data_sparsifier.get_mask(name=valid_name)
        sparsity_level = 1.0 - mask.float().mean()
        logger.info('Sparsity in layer %s = % .2%', name, sparsity_level)