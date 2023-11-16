from recommenders.models.deeprec.deeprec_utils import flat_config, HParams, load_yaml
import random
import re

def check_type(config):
    if False:
        print('Hello World!')
    'Check that the config parameters are the correct type\n\n    Args:\n        config (dict): Configuration dictionary.\n\n    Raises:\n        TypeError: If the parameters are not the correct type.\n    '
    int_parameters = ['word_size', 'his_size', 'title_size', 'body_size', 'npratio', 'word_emb_dim', 'attention_hidden_dim', 'epochs', 'batch_size', 'show_step', 'save_epoch', 'head_num', 'head_dim', 'user_num', 'filter_num', 'window_size', 'gru_unit', 'user_emb_dim', 'vert_emb_dim', 'subvert_emb_dim']
    for param in int_parameters:
        if param in config and (not isinstance(config[param], int)):
            raise TypeError('Parameters {0} must be int'.format(param))
    float_parameters = ['learning_rate', 'dropout']
    for param in float_parameters:
        if param in config and (not isinstance(config[param], float)):
            raise TypeError('Parameters {0} must be float'.format(param))
    str_parameters = ['wordEmb_file', 'wordDict_file', 'userDict_file', 'vertDict_file', 'subvertDict_file', 'method', 'loss', 'optimizer', 'cnn_activation', 'dense_activationtype']
    for param in str_parameters:
        if param in config and (not isinstance(config[param], str)):
            raise TypeError('Parameters {0} must be str'.format(param))
    list_parameters = ['layer_sizes', 'activation']
    for param in list_parameters:
        if param in config and (not isinstance(config[param], list)):
            raise TypeError('Parameters {0} must be list'.format(param))
    bool_parameters = ['support_quick_scoring']
    for param in bool_parameters:
        if param in config and (not isinstance(config[param], bool)):
            raise TypeError('Parameters {0} must be bool'.format(param))

def check_nn_config(f_config):
    if False:
        return 10
    'Check neural networks configuration.\n\n    Args:\n        f_config (dict): Neural network configuration.\n\n    Raises:\n        ValueError: If the parameters are not correct.\n    '
    if f_config['model_type'] in ['nrms', 'NRMS']:
        required_parameters = ['title_size', 'his_size', 'wordEmb_file', 'wordDict_file', 'userDict_file', 'npratio', 'data_format', 'word_emb_dim', 'head_num', 'head_dim', 'attention_hidden_dim', 'loss', 'data_format', 'dropout']
    elif f_config['model_type'] in ['naml', 'NAML']:
        required_parameters = ['title_size', 'body_size', 'his_size', 'wordEmb_file', 'subvertDict_file', 'vertDict_file', 'wordDict_file', 'userDict_file', 'npratio', 'data_format', 'word_emb_dim', 'vert_emb_dim', 'subvert_emb_dim', 'filter_num', 'cnn_activation', 'window_size', 'dense_activation', 'attention_hidden_dim', 'loss', 'data_format', 'dropout']
    elif f_config['model_type'] in ['lstur', 'LSTUR']:
        required_parameters = ['title_size', 'his_size', 'wordEmb_file', 'wordDict_file', 'userDict_file', 'npratio', 'data_format', 'word_emb_dim', 'gru_unit', 'type', 'filter_num', 'cnn_activation', 'window_size', 'attention_hidden_dim', 'loss', 'data_format', 'dropout']
    elif f_config['model_type'] in ['npa', 'NPA']:
        required_parameters = ['title_size', 'his_size', 'wordEmb_file', 'wordDict_file', 'userDict_file', 'npratio', 'data_format', 'word_emb_dim', 'user_emb_dim', 'filter_num', 'cnn_activation', 'window_size', 'attention_hidden_dim', 'loss', 'data_format', 'dropout']
    else:
        required_parameters = []
    for param in required_parameters:
        if param not in f_config:
            raise ValueError('Parameters {0} must be set'.format(param))
    if f_config['model_type'] in ['nrms', 'NRMS', 'lstur', 'LSTUR']:
        if f_config['data_format'] != 'news':
            raise ValueError("For nrms and naml model, data format must be 'news', but your set is {0}".format(f_config['data_format']))
    elif f_config['model_type'] in ['naml', 'NAML']:
        if f_config['data_format'] != 'naml':
            raise ValueError("For nrms and naml model, data format must be 'naml', but your set is {0}".format(f_config['data_format']))
    check_type(f_config)

def create_hparams(flags):
    if False:
        while True:
            i = 10
    'Create the model hyperparameters.\n\n    Args:\n        flags (dict): Dictionary with the model requirements.\n\n    Returns:\n        HParams: Hyperparameter object.\n    '
    init_dict = {'support_quick_scoring': False, 'dropout': 0.0, 'attention_hidden_dim': 200, 'head_num': 4, 'head_dim': 100, 'filter_num': 200, 'window_size': 3, 'vert_emb_dim': 100, 'subvert_emb_dim': 100, 'gru_unit': 400, 'type': 'ini', 'user_emb_dim': 50, 'learning_rate': 0.001, 'optimizer': 'adam', 'epochs': 10, 'batch_size': 1, 'show_step': 1}
    init_dict.update(flags)
    return HParams(init_dict)

def prepare_hparams(yaml_file=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Prepare the model hyperparameters and check that all have the correct value.\n\n    Args:\n        yaml_file (str): YAML file as configuration.\n\n    Returns:\n        HParams: Hyperparameter object.\n    '
    if yaml_file is not None:
        config = load_yaml(yaml_file)
        config = flat_config(config)
    else:
        config = {}
    config.update(kwargs)
    check_nn_config(config)
    return create_hparams(config)

def word_tokenize(sent):
    if False:
        return 10
    'Split sentence into word list using regex.\n    Args:\n        sent (str): Input sentence\n\n    Return:\n        list: word list\n    '
    pat = re.compile('[\\w]+|[.,!?;|]')
    if isinstance(sent, str):
        return pat.findall(sent.lower())
    else:
        return []

def newsample(news, ratio):
    if False:
        while True:
            i = 10
    'Sample ratio samples from news list.\n    If length of news is less than ratio, pad zeros.\n\n    Args:\n        news (list): input news list\n        ratio (int): sample number\n\n    Returns:\n        list: output of sample list.\n    '
    if ratio > len(news):
        return news + [0] * (ratio - len(news))
    else:
        return random.sample(news, ratio)

def get_mind_data_set(type):
    if False:
        i = 10
        return i + 15
    "Get MIND dataset address\n\n    Args:\n        type (str): type of mind dataset, must be in ['large', 'small', 'demo']\n\n    Returns:\n        list: data url and train valid dataset name\n    "
    assert type in ['large', 'small', 'demo']
    if type == 'large':
        return ('https://mind201910small.blob.core.windows.net/release/', 'MINDlarge_train.zip', 'MINDlarge_dev.zip', 'MINDlarge_utils.zip')
    elif type == 'small':
        return ('https://mind201910small.blob.core.windows.net/release/', 'MINDsmall_train.zip', 'MINDsmall_dev.zip', 'MINDsmall_utils.zip')
    elif type == 'demo':
        return ('https://recodatasets.z20.web.core.windows.net/newsrec/', 'MINDdemo_train.zip', 'MINDdemo_dev.zip', 'MINDdemo_utils.zip')