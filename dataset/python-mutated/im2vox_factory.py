"""Factory module for getting the complete image to voxel generation network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from nets import perspective_projector
from nets import ptn_encoder
from nets import ptn_vox_decoder
_NAME_TO_NETS = {'ptn_encoder': ptn_encoder, 'ptn_vox_decoder': ptn_vox_decoder, 'perspective_projector': perspective_projector}

def _get_network(name):
    if False:
        return 10
    'Gets a single encoder/decoder network model.'
    if name not in _NAME_TO_NETS:
        raise ValueError('Network name [%s] not recognized.' % name)
    return _NAME_TO_NETS[name].model

def get(params, is_training=False, reuse=False, run_projection=True):
    if False:
        i = 10
        return i + 15
    'Factory function to get the training/pretraining im->vox model (NIPS16).\n\n  Args:\n    params: Different parameters used througout ptn, typically FLAGS (dict).\n    is_training: Set to True if while training (boolean).\n    reuse: Set as True if sharing variables with a model that has already\n      been built (boolean).\n    run_projection: Set as False if not interested in mask and projection\n      images. Useful in evaluation routine (boolean).\n  Returns:\n    Model function for network (inputs to outputs).\n  '

    def model(inputs):
        if False:
            while True:
                i = 10
        'Model function corresponding to a specific network architecture.'
        outputs = {}
        encoder_fn = _get_network(params.encoder_name)
        with tf.variable_scope('encoder', reuse=reuse):
            enc_outputs = encoder_fn(inputs['images_1'], params, is_training)
            outputs['ids_1'] = enc_outputs['ids']
        decoder_fn = _get_network(params.decoder_name)
        with tf.variable_scope('decoder', reuse=reuse):
            outputs['voxels_1'] = decoder_fn(outputs['ids_1'], params, is_training)
        if run_projection:
            projector_fn = _get_network(params.projector_name)
            with tf.variable_scope('projector', reuse=reuse):
                outputs['projs_1'] = projector_fn(outputs['voxels_1'], inputs['matrix_1'], params, is_training)
            with tf.variable_scope('oracle', reuse=reuse):
                outputs['masks_1'] = projector_fn(inputs['voxels'], inputs['matrix_1'], params, False)
            for k in range(1, params.step_size):
                with tf.variable_scope('projector', reuse=True):
                    outputs['projs_%d' % (k + 1)] = projector_fn(outputs['voxels_1'], inputs['matrix_%d' % (k + 1)], params, is_training)
                with tf.variable_scope('oracle', reuse=True):
                    outputs['masks_%d' % (k + 1)] = projector_fn(inputs['voxels'], inputs['matrix_%d' % (k + 1)], params, False)
        return outputs
    return model