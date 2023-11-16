"""Factory module for different encoder/decoder network models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from nets import ptn_encoder
from nets import ptn_im_decoder
from nets import ptn_rotator
_NAME_TO_NETS = {'ptn_encoder': ptn_encoder, 'ptn_rotator': ptn_rotator, 'ptn_im_decoder': ptn_im_decoder}

def _get_network(name):
    if False:
        print('Hello World!')
    'Gets a single network component.'
    if name not in _NAME_TO_NETS:
        raise ValueError('Network name [%s] not recognized.' % name)
    return _NAME_TO_NETS[name].model

def get(params, is_training=False, reuse=False):
    if False:
        i = 10
        return i + 15
    'Factory function to retrieve a network model.\n\n  Args:\n    params: Different parameters used througout ptn, typically FLAGS (dict)\n    is_training: Set to True if while training (boolean)\n    reuse: Set as True if either using a pre-trained model or\n      in the training loop while the graph has already been built (boolean)\n  Returns:\n    Model function for network (inputs to outputs)\n  '

    def model(inputs):
        if False:
            i = 10
            return i + 15
        'Model function corresponding to a specific network architecture.'
        outputs = {}
        encoder_fn = _get_network(params.encoder_name)
        with tf.variable_scope('encoder', reuse=reuse):
            features = encoder_fn(inputs['images_0'], params, is_training)
            outputs['ids'] = features['ids']
            outputs['poses_0'] = features['poses']
        rotator_fn = _get_network(params.rotator_name)
        with tf.variable_scope('rotator', reuse=reuse):
            outputs['poses_1'] = rotator_fn(outputs['poses_0'], inputs['actions'], params, is_training)
        decoder_fn = _get_network(params.decoder_name)
        with tf.variable_scope('decoder', reuse=reuse):
            dec_output = decoder_fn(outputs['ids'], outputs['poses_1'], params, is_training)
            outputs['images_1'] = dec_output['images']
            outputs['masks_1'] = dec_output['masks']
        for k in range(1, params.step_size):
            with tf.variable_scope('rotator', reuse=True):
                outputs['poses_%d' % (k + 1)] = rotator_fn(outputs['poses_%d' % k], inputs['actions'], params, is_training)
            with tf.variable_scope('decoder', reuse=True):
                dec_output = decoder_fn(outputs['ids'], outputs['poses_%d' % (k + 1)], params, is_training)
                outputs['images_%d' % (k + 1)] = dec_output['images']
                outputs['masks_%d' % (k + 1)] = dec_output['masks']
        return outputs
    return model