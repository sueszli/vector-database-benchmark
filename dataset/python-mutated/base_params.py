def get_base_params(dataset_name, checkpoint_dir):
    if False:
        while True:
            i = 10
    params = {}
    basic_params = {'dataset': dataset_name, 'dataroot': './data/' + dataset_name, 'name': '', 'gpu_ids': [0], 'checkpoints_dir': checkpoint_dir, 'verbose': False, 'print_freq': 100}
    params.update(basic_params)
    dataset_params = {'dataset_mode': 'aligned', 'direction': 'BtoA', 'num_threads': 4, 'max_dataset_size': float('inf'), 'preprocess': 'resize_and_crop', 'display_winsize': 256, 'input_nc': 3, 'output_nc': 3}
    params.update(dataset_params)
    model_params = {'model': 'pix2pix', 'n_layers_D': 3, 'init_gain': 0.02, 'no_dropout': False}
    params.update(model_params)
    train_params = {'phase': 'train', 'isTrain': True, 'serial_batches': False, 'load_size': 286, 'crop_size': 256, 'no_flip': False, 'pool_size': 0, 'lr_decay_iters': 50, 'epoch_count': 1, 'continue_train': False}
    train_params.update(params)
    test_params = {'phase': 'test', 'isTrain': False, 'load_iter': -1, 'epoch': 'latest', 'load_size': 256, 'crop_size': 256, 'serial_batches': True, 'no_flip': True, 'eval': True}
    test_params.update(params)
    return (train_params, test_params)