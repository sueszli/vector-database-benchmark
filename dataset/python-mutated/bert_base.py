from bigdl.orca.tfpark.estimator import *
from bert import modeling
from bigdl.dllib.utils.log4Error import *

def bert_model(features, labels, mode, params):
    if False:
        print('Hello World!')
    '\n    Return an instance of BertModel and one can take its different outputs to\n    perform specific tasks.\n    '
    import tensorflow as tf
    input_ids = features['input_ids']
    if 'input_mask' in features:
        input_mask = features['input_mask']
    else:
        input_mask = None
    if 'token_type_ids' in features:
        token_type_ids = features['token_type_ids']
    else:
        token_type_ids = None
    bert_config = modeling.BertConfig.from_json_file(params['bert_config_file'])
    model = modeling.BertModel(config=bert_config, is_training=mode == tf.estimator.ModeKeys.TRAIN, input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids, use_one_hot_embeddings=params['use_one_hot_embeddings'])
    tvars = tf.trainable_variables()
    if params['init_checkpoint']:
        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, params['init_checkpoint'])
        tf.train.init_from_checkpoint(params['init_checkpoint'], assignment_map)
    return model

def bert_input_fn(rdd, max_seq_length, batch_size, features={'input_ids', 'input_mask', 'token_type_ids'}, extra_features=None, labels=None, label_size=None):
    if False:
        print('Hello World!')
    '\n    Takes an RDD to create the input function for BERT related TFEstimators.\n    For training and evaluation, each element in rdd should be a tuple:\n    (dict of features, a single label or dict of labels)\n    Note that currently only integer or integer array labels are supported.\n    For prediction, each element in rdd should be a dict of features.\n\n    Features in each RDD element should contain "input_ids", "input_mask" and "token_type_ids",\n    each of shape max_seq_length.\n    If you have other extra features in your dict of features, you need to explicitly specify\n    the argument `extra_features`, which is supposed to be the dict with feature name as key\n    and tuple of (dtype, shape) as its value.\n    '
    import tensorflow as tf
    invalidInputError(features.issubset({'input_ids', 'input_mask', 'token_type_ids'}), 'features should be subset of {input_ids, input_mask, token_type_ids}')
    features_dict = {}
    for feature in features:
        features_dict[feature] = (tf.int32, [max_seq_length])
    if extra_features is not None:
        invalidInputError(isinstance(extra_features, dict), 'extra_features should be a dictionary')
        for (k, v) in extra_features.items():
            invalidInputError(isinstance(k, six.string_types, 'expect k is string type'))
            invalidInputError(isinstance(v, tuple), 'expect v is tuple')
            features_dict[k] = v
    if label_size is None:
        label_size = []
    else:
        label_size = [label_size]
    if labels is None:
        res_labels = (tf.int32, label_size)
    elif isinstance(labels, list) or isinstance(labels, set):
        labels = set(labels)
        if len(labels) == 1:
            res_labels = (tf.int32, label_size)
        else:
            res_labels = {}
            for label in labels:
                res_labels[label] = (tf.int32, label_size)
    else:
        invalidInputError(False, 'Wrong labels. labels should be a set of label names if you have multiple labels')

    def input_fn(mode):
        if False:
            return 10
        if mode == tf.estimator.ModeKeys.TRAIN:
            return TFDataset.from_rdd(rdd, features=features_dict, labels=res_labels, batch_size=batch_size)
        elif mode == tf.estimator.ModeKeys.EVAL:
            return TFDataset.from_rdd(rdd, features=features_dict, labels=res_labels, batch_per_thread=batch_size // rdd.getNumPartitions())
        else:
            return TFDataset.from_rdd(rdd, features=features_dict, batch_per_thread=batch_size // rdd.getNumPartitions())
    return input_fn

class BERTBaseEstimator(TFEstimator):
    """
    The base class for BERT related TFEstimators.
    Common arguments:
    bert_config_file, init_checkpoint, use_one_hot_embeddings, optimizer, model_dir.

    For its subclass:
    - One can add additional arguments and access them via `params`.
    - One can utilize `_bert_model` to create model_fn and `bert_input_fn` to create input_fn.
    """

    def __init__(self, model_fn, bert_config_file, init_checkpoint=None, use_one_hot_embeddings=False, model_dir=None, **kwargs):
        if False:
            print('Hello World!')
        import tensorflow as tf
        params = {'bert_config_file': bert_config_file, 'init_checkpoint': init_checkpoint, 'use_one_hot_embeddings': use_one_hot_embeddings}
        for (k, v) in kwargs.items():
            params[k] = v
        estimator = tf.estimator.Estimator(model_fn, model_dir=model_dir, params=params)
        super(BERTBaseEstimator, self).__init__(estimator)