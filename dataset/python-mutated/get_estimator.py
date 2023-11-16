"""Get a configured estimator."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from estimators import mvtcn_estimator as mvtcn_estimators
from estimators import svtcn_estimator

def get_mvtcn_estimator(loss_strategy, config, logdir):
    if False:
        for i in range(10):
            print('nop')
    'Returns a configured MVTCN estimator.'
    loss_to_trainer = {'triplet_semihard': mvtcn_estimators.MVTCNTripletEstimator, 'npairs': mvtcn_estimators.MVTCNNpairsEstimator}
    if loss_strategy not in loss_to_trainer:
        raise ValueError('Unknown loss for MVTCN: %s' % loss_strategy)
    estimator = loss_to_trainer[loss_strategy](config, logdir)
    return estimator

def get_estimator(config, logdir):
    if False:
        return 10
    'Returns an unsupervised model trainer based on config.\n\n  Args:\n    config: A T object holding training configs.\n    logdir: String, path to directory where model checkpoints and summaries\n      are saved.\n  Returns:\n    estimator: A configured `TCNEstimator` object.\n  Raises:\n    ValueError: If unknown training strategy is specified.\n  '
    training_strategy = config.training_strategy
    if training_strategy == 'mvtcn':
        loss_strategy = config.loss_strategy
        estimator = get_mvtcn_estimator(loss_strategy, config, logdir)
    elif training_strategy == 'svtcn':
        estimator = svtcn_estimator.SVTCNTripletEstimator(config, logdir)
    else:
        raise ValueError('Unknown training strategy: %s' % training_strategy)
    return estimator