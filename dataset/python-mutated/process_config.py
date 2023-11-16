def process_config(ludwig_config: dict, experiment_dict: dict) -> dict:
    if False:
        for i in range(10):
            print('nop')
    'Modify a Ludwig config.\n\n    :param ludwig_config: a Ludwig config.\n    :param experiment_dict: a benchmarking config experiment dictionary.\n\n    returns: a modified Ludwig config.\n    '
    if experiment_dict['dataset_name'] == 'ames_housing':
        main_config_keys = list(ludwig_config.keys())
        for key in main_config_keys:
            if key not in ['input_features', 'output_features']:
                del ludwig_config[key]
    ludwig_config['trainer'] = {'early_stop': 7}
    ludwig_config['combiner'] = {'type': 'concat'}
    for (i, feature) in enumerate(ludwig_config['input_features']):
        if feature['type'] == 'category':
            ludwig_config['input_features'][i]['encoder'] = 'sparse'
    for (i, feature) in enumerate(ludwig_config['output_features']):
        if feature['type'] == 'category':
            ludwig_config['output_features'][i]['encoder'] = 'sparse'
    return ludwig_config