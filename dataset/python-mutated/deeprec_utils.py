import os
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error, accuracy_score, f1_score
import numpy as np
import yaml
import zipfile
import pickle as pkl
from recommenders.datasets.download_utils import maybe_download

def flat_config(config):
    if False:
        while True:
            i = 10
    'Flat config loaded from a yaml file to a flat dict.\n\n    Args:\n        config (dict): Configuration loaded from a yaml file.\n\n    Returns:\n        dict: Configuration dictionary.\n    '
    f_config = {}
    category = config.keys()
    for cate in category:
        for (key, val) in config[cate].items():
            f_config[key] = val
    return f_config

def check_type(config):
    if False:
        while True:
            i = 10
    'Check that the config parameters are the correct type\n\n    Args:\n        config (dict): Configuration dictionary.\n\n    Raises:\n        TypeError: If the parameters are not the correct type.\n    '
    int_parameters = ['word_size', 'entity_size', 'doc_size', 'history_size', 'FEATURE_COUNT', 'FIELD_COUNT', 'dim', 'epochs', 'batch_size', 'show_step', 'save_epoch', 'PAIR_NUM', 'DNN_FIELD_NUM', 'attention_layer_sizes', 'n_user', 'n_item', 'n_user_attr', 'n_item_attr', 'item_embedding_dim', 'cate_embedding_dim', 'user_embedding_dim', 'max_seq_length', 'hidden_size', 'T', 'L', 'n_v', 'n_h', 'kernel_size', 'min_seq_length', 'attention_size', 'epochs', 'batch_size', 'show_step', 'save_epoch', 'train_num_ngs']
    for param in int_parameters:
        if param in config and (not isinstance(config[param], int)):
            raise TypeError('Parameters {0} must be int'.format(param))
    float_parameters = ['init_value', 'learning_rate', 'embed_l2', 'embed_l1', 'layer_l2', 'layer_l1', 'mu']
    for param in float_parameters:
        if param in config and (not isinstance(config[param], float)):
            raise TypeError('Parameters {0} must be float'.format(param))
    str_parameters = ['train_file', 'eval_file', 'test_file', 'infer_file', 'method', 'load_model_name', 'infer_model_name', 'loss', 'optimizer', 'init_method', 'attention_activation', 'user_vocab', 'item_vocab', 'cate_vocab']
    for param in str_parameters:
        if param in config and (not isinstance(config[param], str)):
            raise TypeError('Parameters {0} must be str'.format(param))
    list_parameters = ['layer_sizes', 'activation', 'dropout', 'att_fcn_layer_sizes', 'dilations']
    for param in list_parameters:
        if param in config and (not isinstance(config[param], list)):
            raise TypeError('Parameters {0} must be list'.format(param))

def check_nn_config(f_config):
    if False:
        return 10
    'Check neural networks configuration.\n\n    Args:\n        f_config (dict): Neural network configuration.\n\n    Raises:\n        ValueError: If the parameters are not correct.\n    '
    if f_config['model_type'] in ['fm', 'FM']:
        required_parameters = ['FEATURE_COUNT', 'dim', 'loss', 'data_format', 'method']
    elif f_config['model_type'] in ['lr', 'LR']:
        required_parameters = ['FEATURE_COUNT', 'loss', 'data_format', 'method']
    elif f_config['model_type'] in ['dkn', 'DKN']:
        required_parameters = ['doc_size', 'history_size', 'wordEmb_file', 'entityEmb_file', 'contextEmb_file', 'news_feature_file', 'user_history_file', 'word_size', 'entity_size', 'use_entity', 'use_context', 'data_format', 'dim', 'layer_sizes', 'activation', 'attention_activation', 'attention_activation', 'attention_dropout', 'loss', 'data_format', 'dropout', 'method', 'num_filters', 'filter_sizes']
    elif f_config['model_type'] in ['exDeepFM', 'xDeepFM']:
        required_parameters = ['FIELD_COUNT', 'FEATURE_COUNT', 'method', 'dim', 'layer_sizes', 'cross_layer_sizes', 'activation', 'loss', 'data_format', 'dropout']
    if f_config['model_type'] in ['gru', 'GRU']:
        required_parameters = ['item_embedding_dim', 'cate_embedding_dim', 'max_seq_length', 'loss', 'method', 'user_vocab', 'item_vocab', 'cate_vocab', 'hidden_size']
    elif f_config['model_type'] in ['caser', 'CASER', 'Caser']:
        required_parameters = ['item_embedding_dim', 'cate_embedding_dim', 'user_embedding_dim', 'max_seq_length', 'loss', 'method', 'user_vocab', 'item_vocab', 'cate_vocab', 'T', 'L', 'n_v', 'n_h', 'min_seq_length']
    elif f_config['model_type'] in ['asvd', 'ASVD', 'a2svd', 'A2SVD']:
        required_parameters = ['item_embedding_dim', 'cate_embedding_dim', 'max_seq_length', 'loss', 'method', 'user_vocab', 'item_vocab', 'cate_vocab']
    elif f_config['model_type'] in ['slirec', 'sli_rec', 'SLI_REC', 'Sli_rec']:
        required_parameters = ['item_embedding_dim', 'cate_embedding_dim', 'max_seq_length', 'loss', 'method', 'user_vocab', 'item_vocab', 'cate_vocab', 'attention_size', 'hidden_size', 'att_fcn_layer_sizes']
    elif f_config['model_type'] in ['nextitnet', 'next_it_net', 'NextItNet', 'NEXT_IT_NET']:
        required_parameters = ['item_embedding_dim', 'cate_embedding_dim', 'user_embedding_dim', 'max_seq_length', 'loss', 'method', 'user_vocab', 'item_vocab', 'cate_vocab', 'dilations', 'kernel_size', 'min_seq_length']
    else:
        required_parameters = []
    for param in required_parameters:
        if param not in f_config:
            raise ValueError('Parameters {0} must be set'.format(param))
    if f_config['model_type'] in ['exDeepFM', 'xDeepFM']:
        if f_config['data_format'] != 'ffm':
            raise ValueError("For xDeepFM model, data format must be 'ffm', but your set is {0}".format(f_config['data_format']))
    elif f_config['model_type'] in ['dkn', 'DKN']:
        if f_config['data_format'] != 'dkn':
            raise ValueError("For dkn model, data format must be 'dkn', but your set is {0}".format(f_config['data_format']))
    check_type(f_config)

def load_yaml(filename):
    if False:
        for i in range(10):
            print('nop')
    'Load a yaml file.\n\n    Args:\n        filename (str): Filename.\n\n    Returns:\n        dict: Dictionary.\n    '
    try:
        with open(filename, 'r') as f:
            config = yaml.load(f, yaml.SafeLoader)
        return config
    except FileNotFoundError:
        raise
    except Exception:
        raise IOError('load {0} error!'.format(filename))

class HParams:
    """Class for holding hyperparameters for DeepRec algorithms."""

    def __init__(self, hparams_dict):
        if False:
            print('Hello World!')
        'Create an HParams object from a dictionary of hyperparameter values.\n\n        Args:\n            hparams_dict (dict): Dictionary with the model hyperparameters.\n        '
        for val in hparams_dict.values():
            if not (isinstance(val, int) or isinstance(val, float) or isinstance(val, str) or isinstance(val, list)):
                raise ValueError('Hyperparameter value {} should be integer, float, string or list.'.format(val))
        self._values = hparams_dict
        for hparam in hparams_dict:
            setattr(self, hparam, hparams_dict[hparam])

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return 'HParams object with values {}'.format(self._values.__repr__())

    def values(self):
        if False:
            i = 10
            return i + 15
        'Return the hyperparameter values as a dictionary.\n\n        Returns:\n            dict: Dictionary with the hyperparameter values.\n        '
        return self._values

def create_hparams(flags):
    if False:
        for i in range(10):
            print('nop')
    'Create the model hyperparameters.\n\n    Args:\n        flags (dict): Dictionary with the model requirements.\n\n    Returns:\n        HParams: Hyperparameter object.\n    '
    init_dict = {'use_entity': True, 'use_context': True, 'cross_activation': 'identity', 'user_dropout': False, 'dropout': [0.0], 'attention_dropout': 0.0, 'load_saved_model': False, 'fast_CIN_d': 0, 'use_Linear_part': False, 'use_FM_part': False, 'use_CIN_part': False, 'use_DNN_part': False, 'init_method': 'tnormal', 'init_value': 0.01, 'embed_l2': 0.0, 'embed_l1': 0.0, 'layer_l2': 0.0, 'layer_l1': 0.0, 'cross_l2': 0.0, 'cross_l1': 0.0, 'reg_kg': 0.0, 'learning_rate': 0.001, 'lr_rs': 1, 'lr_kg': 0.5, 'kg_training_interval': 5, 'max_grad_norm': 2, 'is_clip_norm': 0, 'dtype': 32, 'optimizer': 'adam', 'epochs': 10, 'batch_size': 1, 'enable_BN': False, 'show_step': 1, 'save_model': True, 'save_epoch': 5, 'write_tfevents': False, 'train_num_ngs': 4, 'need_sample': True, 'embedding_dropout': 0.0, 'EARLY_STOP': 100, 'min_seq_length': 1, 'slots': 5, 'cell': 'SUM'}
    init_dict.update(flags)
    return HParams(init_dict)

def prepare_hparams(yaml_file=None, **kwargs):
    if False:
        while True:
            i = 10
    'Prepare the model hyperparameters and check that all have the correct value.\n\n    Args:\n        yaml_file (str): YAML file as configuration.\n\n    Returns:\n        HParams: Hyperparameter object.\n    '
    if yaml_file is not None:
        config = load_yaml(yaml_file)
        config = flat_config(config)
    else:
        config = {}
    if kwargs:
        for (name, value) in kwargs.items():
            config[name] = value
    check_nn_config(config)
    return create_hparams(config)

def download_deeprec_resources(azure_container_url, data_path, remote_resource_name):
    if False:
        for i in range(10):
            print('nop')
    'Download resources.\n\n    Args:\n        azure_container_url (str): URL of Azure container.\n        data_path (str): Path to download the resources.\n        remote_resource_name (str): Name of the resource.\n    '
    os.makedirs(data_path, exist_ok=True)
    remote_path = azure_container_url + remote_resource_name
    maybe_download(remote_path, remote_resource_name, data_path)
    zip_ref = zipfile.ZipFile(os.path.join(data_path, remote_resource_name), 'r')
    zip_ref.extractall(data_path)
    zip_ref.close()
    os.remove(os.path.join(data_path, remote_resource_name))

def mrr_score(y_true, y_score):
    if False:
        print('Hello World!')
    'Computing mrr score metric.\n\n    Args:\n        y_true (np.ndarray): Ground-truth labels.\n        y_score (np.ndarray): Predicted labels.\n\n    Returns:\n        numpy.ndarray: mrr scores.\n    '
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)

def ndcg_score(y_true, y_score, k=10):
    if False:
        return 10
    'Computing ndcg score metric at k.\n\n    Args:\n        y_true (np.ndarray): Ground-truth labels.\n        y_score (np.ndarray): Predicted labels.\n\n    Returns:\n        numpy.ndarray: ndcg scores.\n    '
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best

def hit_score(y_true, y_score, k=10):
    if False:
        while True:
            i = 10
    'Computing hit score metric at k.\n\n    Args:\n        y_true (np.ndarray): ground-truth labels.\n        y_score (np.ndarray): predicted labels.\n\n    Returns:\n        np.ndarray: hit score.\n    '
    ground_truth = np.where(y_true == 1)[0]
    argsort = np.argsort(y_score)[::-1][:k]
    for idx in argsort:
        if idx in ground_truth:
            return 1
    return 0

def dcg_score(y_true, y_score, k=10):
    if False:
        return 10
    'Computing dcg score metric at k.\n\n    Args:\n        y_true (np.ndarray): Ground-truth labels.\n        y_score (np.ndarray): Predicted labels.\n\n    Returns:\n        np.ndarray: dcg scores.\n    '
    k = min(np.shape(y_true)[-1], k)
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def cal_metric(labels, preds, metrics):
    if False:
        return 10
    'Calculate metrics.\n\n    Available options are: `auc`, `rmse`, `logloss`, `acc` (accurary), `f1`, `mean_mrr`,\n    `ndcg` (format like: ndcg@2;4;6;8), `hit` (format like: hit@2;4;6;8), `group_auc`.\n\n    Args:\n        labels (array-like): Labels.\n        preds (array-like): Predictions.\n        metrics (list): List of metric names.\n\n    Return:\n        dict: Metrics.\n\n    Examples:\n        >>> cal_metric(labels, preds, ["ndcg@2;4;6", "group_auc"])\n        {\'ndcg@2\': 0.4026, \'ndcg@4\': 0.4953, \'ndcg@6\': 0.5346, \'group_auc\': 0.8096}\n\n    '
    res = {}
    for metric in metrics:
        if metric == 'auc':
            auc = roc_auc_score(np.asarray(labels), np.asarray(preds))
            res['auc'] = round(auc, 4)
        elif metric == 'rmse':
            rmse = mean_squared_error(np.asarray(labels), np.asarray(preds))
            res['rmse'] = np.sqrt(round(rmse, 4))
        elif metric == 'logloss':
            preds = [max(min(p, 1.0 - 1e-11), 1e-11) for p in preds]
            logloss = log_loss(np.asarray(labels), np.asarray(preds))
            res['logloss'] = round(logloss, 4)
        elif metric == 'acc':
            pred = np.asarray(preds)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            acc = accuracy_score(np.asarray(labels), pred)
            res['acc'] = round(acc, 4)
        elif metric == 'f1':
            pred = np.asarray(preds)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            f1 = f1_score(np.asarray(labels), pred)
            res['f1'] = round(f1, 4)
        elif metric == 'mean_mrr':
            mean_mrr = np.mean([mrr_score(each_labels, each_preds) for (each_labels, each_preds) in zip(labels, preds)])
            res['mean_mrr'] = round(mean_mrr, 4)
        elif metric.startswith('ndcg'):
            ndcg_list = [1, 2]
            ks = metric.split('@')
            if len(ks) > 1:
                ndcg_list = [int(token) for token in ks[1].split(';')]
            for k in ndcg_list:
                ndcg_temp = np.mean([ndcg_score(each_labels, each_preds, k) for (each_labels, each_preds) in zip(labels, preds)])
                res['ndcg@{0}'.format(k)] = round(ndcg_temp, 4)
        elif metric.startswith('hit'):
            hit_list = [1, 2]
            ks = metric.split('@')
            if len(ks) > 1:
                hit_list = [int(token) for token in ks[1].split(';')]
            for k in hit_list:
                hit_temp = np.mean([hit_score(each_labels, each_preds, k) for (each_labels, each_preds) in zip(labels, preds)])
                res['hit@{0}'.format(k)] = round(hit_temp, 4)
        elif metric == 'group_auc':
            group_auc = np.mean([roc_auc_score(each_labels, each_preds) for (each_labels, each_preds) in zip(labels, preds)])
            res['group_auc'] = round(group_auc, 4)
        else:
            raise ValueError('Metric {0} not defined'.format(metric))
    return res

def load_dict(filename):
    if False:
        print('Hello World!')
    'Load the vocabularies.\n\n    Args:\n        filename (str): Filename of user, item or category vocabulary.\n\n    Returns:\n        dict: A saved vocabulary.\n    '
    with open(filename, 'rb') as f:
        f_pkl = pkl.load(f)
        return f_pkl