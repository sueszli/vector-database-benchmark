import logging
from neon.models.model import Model
from neon.callbacks.callbacks import Callbacks
from neon.util.persist import load_class, load_obj
logger = logging.getLogger(__name__)

def deserialize(fn, datasets=None, inference=False):
    if False:
        print('Hello World!')
    '\n    Helper function to load all objects from a serialized file,\n    this includes callbacks and datasets as well as the model, layers,\n    etc.\n\n    Arguments:\n        datasets (DataSet, optional): If the dataset is not serialized\n                                      in the file it can be passed in\n                                      as an argument.  This will also\n                                      override any dataset in the serialized\n                                      file\n        inference (bool, optional): if true only the weights will be loaded, not\n                                    the states\n    Returns:\n        Model: the model object\n        Dataset: the data set object\n        Callback: the callbacks\n    '
    config_dict = load_obj(fn)
    if datasets is not None:
        logger.warn('Ignoring datasets serialized in archive file %s' % fn)
    elif 'datasets' in config_dict:
        ds_cls = load_class(config_dict['datasets']['type'])
        dataset = ds_cls.gen_class(config_dict['datasets']['config'])
        datasets = dataset.gen_iterators()
    if 'train' in datasets:
        data_iter = datasets['train']
    else:
        key = list(datasets.keys())[0]
        data_iter = datasets[key]
        logger.warn('Could not find training set iteratorusing %s instead' % key)
    model = Model(config_dict, data_iter)
    callbacks = None
    if 'callbacks' in config_dict:
        cbs = config_dict['callbacks']['callbacks']
        for cb in cbs:
            if 'config' not in cb:
                cb['config'] = {}
            for arg in cb['config']:
                if type(cb['config'][arg]) is dict and 'type' in cb['config'][arg]:
                    if cb['config'][arg]['type'] == 'Data':
                        key = cb['config'][arg]['name']
                        if key in datasets:
                            cb['config'][arg] = datasets[key]
                        else:
                            cb['config'][arg] = None
        callbacks = Callbacks.load_callbacks(config_dict['callbacks'], model)
    return (model, dataset, callbacks)