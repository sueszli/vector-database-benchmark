import importlib
import random
from collections import OrderedDict
import numpy as np
import torch
from modelscope.models.cv.video_depth_estimation.utils.load import filter_args, load_class, load_class_args_create, load_network
from modelscope.models.cv.video_depth_estimation.utils.misc import pcolor

class ModelWrapper(torch.nn.Module):
    """
    Top-level torch.nn.Module wrapper around a SfmModel (pose+depth networks).
    Designed to use models with high-level Trainer classes (cf. trainers/).

    Parameters
    ----------
    config : CfgNode
        Model configuration (cf. configs/default_config.py)
    """

    def __init__(self, config, resume=None, logger=None, load_datasets=True):
        if False:
            return 10
        super().__init__()
        self.config = config
        self.logger = logger
        self.resume = resume
        set_random_seed(config.arch.seed)
        self.metrics_name = 'depth'
        self.metrics_keys = ('abs_rel', 'sqr_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3', 'SILog', 'l1_inv', 'rot_ang', 't_ang', 't_cm')
        self.metrics_modes = ('', '_pp', '_gt', '_pp_gt')
        self.model = self.optimizer = self.scheduler = None
        self.train_dataset = self.validation_dataset = self.test_dataset = None
        self.current_epoch = 0
        self.prepare_model(resume)
        self.config.prepared = True

    def prepare_model(self, resume=None):
        if False:
            while True:
                i = 10
        'Prepare self.model (incl. loading previous state)'
        print0(pcolor('### Preparing Model', 'green'))
        self.model = setup_model(self.config.model, self.config.prepared)
        if resume:
            print0(pcolor('### Resuming from {}'.format(resume['file']), 'magenta', attrs=['bold']))
            self.model = load_network(self.model, resume['state_dict'], 'model')
            if 'epoch' in resume:
                self.current_epoch = resume['epoch']

    @property
    def depth_net(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns depth network.'
        return self.model.depth_net

    @property
    def pose_net(self):
        if False:
            i = 10
            return i + 15
        'Returns pose network.'
        return self.model.pose_net

    @property
    def percep_net(self):
        if False:
            i = 10
            return i + 15
        'Returns perceptual network.'
        return self.model.percep_net

    @property
    def logs(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns various logs for tracking.'
        params = OrderedDict()
        for param in self.optimizer.param_groups:
            params['{}_learning_rate'.format(param['name'].lower())] = param['lr']
        params['progress'] = self.progress
        return {**params, **self.model.logs}

    @property
    def progress(self):
        if False:
            print('Hello World!')
        'Returns training progress (current epoch / max. number of epochs)'
        return self.current_epoch / self.config.arch.max_epochs

    def configure_optimizers(self):
        if False:
            print('Hello World!')
        'Configure depth and pose optimizers and the corresponding scheduler.'
        params = []
        optimizer = getattr(torch.optim, self.config.model.optimizer.name)
        if self.depth_net is not None:
            params.append({'name': 'Depth', 'params': self.depth_net.parameters(), **filter_args(optimizer, self.config.model.optimizer.depth)})
        if self.pose_net is not None:
            params.append({'name': 'Pose', 'params': [param for param in self.pose_net.parameters() if param.requires_grad], **filter_args(optimizer, self.config.model.optimizer.pose)})
        optimizer = optimizer(params)
        scheduler = getattr(torch.optim.lr_scheduler, self.config.model.scheduler.name)
        scheduler = scheduler(optimizer, **filter_args(scheduler, self.config.model.scheduler))
        self.optimizer = optimizer
        self.scheduler = scheduler
        return (optimizer, scheduler)

    def forward(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        'Runs the model and returns the output.'
        assert self.model is not None, 'Model not defined'
        return self.model(*args, **kwargs)

    def depth(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Runs the pose network and returns the output.'
        assert self.depth_net is not None, 'Depth network not defined'
        return self.depth_net(*args, **kwargs)

    def pose(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Runs the depth network and returns the output.'
        assert self.pose_net is not None, 'Pose network not defined'
        return self.pose_net(*args, **kwargs)

    def percep(self, *args, **kwargs):
        if False:
            return 10
        'Runs the depth network and returns the output.'
        assert self.percep_net is not None, 'Perceptual network not defined'
        return self.percep_net(*args, **kwargs)

def set_random_seed(seed):
    if False:
        i = 10
        return i + 15
    if seed >= 0:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def setup_depth_net(config, prepared, **kwargs):
    if False:
        while True:
            i = 10
    '\n    Create a depth network\n\n    Parameters\n    ----------\n    config : CfgNode\n        Network configuration\n    prepared : bool\n        True if the network has been prepared before\n    kwargs : dict\n        Extra parameters for the network\n\n    Returns\n    -------\n    depth_net : nn.Module\n        Create depth network\n    '
    print0(pcolor('DepthNet: %s' % config.name, 'yellow'))
    if config.name == 'DepthPoseNet':
        model_class = getattr(importlib.import_module('modelscope.models.cv.video_depth_estimation.networks.depth_pose.depth_pose_net'), 'DepthPoseNet')
    depth_net = model_class(**{**config, **kwargs})
    if not prepared and config.checkpoint_path != '':
        depth_net = load_network(depth_net, config.checkpoint_path, ['depth_net', 'disp_network'])
    return depth_net

def setup_pose_net(config, prepared, **kwargs):
    if False:
        return 10
    '\n    Create a pose network\n\n    Parameters\n    ----------\n    config : CfgNode\n        Network configuration\n    prepared : bool\n        True if the network has been prepared before\n    kwargs : dict\n        Extra parameters for the network\n\n    Returns\n    -------\n    pose_net : nn.Module\n        Created pose network\n    '
    print0(pcolor('PoseNet: %s' % config.name, 'yellow'))
    pose_net = load_class_args_create(config.name, paths=['modelscope.models.cv.video_depth_estimation.networks.pose'], args={**config, **kwargs})
    if not prepared and config.checkpoint_path != '':
        pose_net = load_network(pose_net, config.checkpoint_path, ['pose_net', 'pose_network'])
    return pose_net

def setup_percep_net(config, prepared, **kwargs):
    if False:
        print('Hello World!')
    '\n    Create a perceputal network\n\n    Parameters\n    ----------\n    config : CfgNode\n        Network configuration\n    prepared : bool\n        True if the network has been prepared before\n    kwargs : dict\n        Extra parameters for the network\n\n    Returns\n    -------\n    depth_net : nn.Module\n        Create depth network\n    '
    print0(pcolor('PercepNet: %s' % config.name, 'yellow'))
    percep_net = load_class_args_create(config.name, paths=['modelscope.models.cv.video_depth_estimation.networks.layers'], args={**config, **kwargs})
    return percep_net

def setup_model(config, prepared, **kwargs):
    if False:
        print('Hello World!')
    '\n    Create a model\n\n    Parameters\n    ----------\n    config : CfgNode\n        Model configuration (cf. configs/default_config.py)\n    prepared : bool\n        True if the model has been prepared before\n    kwargs : dict\n        Extra parameters for the model\n\n    Returns\n    -------\n    model : nn.Module\n        Created model\n    '
    print0(pcolor('Model: %s' % config.name, 'yellow'))
    config.loss.min_depth = config.params.min_depth
    config.loss.max_depth = config.params.max_depth
    if config.name == 'SupModelMF':
        model_class = getattr(importlib.import_module('modelscope.models.cv.video_depth_estimation.models.sup_model_mf'), 'SupModelMF')
    model = model_class(**{**config.loss, **kwargs})
    if model.network_requirements['depth_net']:
        config.depth_net.max_depth = config.params.max_depth
        config.depth_net.min_depth = config.params.min_depth
        model.add_depth_net(setup_depth_net(config.depth_net, prepared))
    if model.network_requirements['pose_net']:
        model.add_pose_net(setup_pose_net(config.pose_net, prepared))
    if model.network_requirements['percep_net']:
        model.add_percep_net(setup_percep_net(config.percep_net, prepared))
    if not prepared and config.checkpoint_path != '':
        model = load_network(model, config.checkpoint_path, 'model')
    return model

def print0(string='\n'):
    if False:
        print('Hello World!')
    print(string)