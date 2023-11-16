import numpy as np
import os
import ray
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.framework import try_import_tf
from ray.tune.registry import get_trainable_cls
(tf1, tf, tfv) = try_import_tf()
ray.init()

def train_and_export_policy_and_model(algo_name, num_steps, model_dir, ckpt_dir):
    if False:
        i = 10
        return i + 15
    cls = get_trainable_cls(algo_name)
    config = cls.get_default_config()
    config.framework('tf')
    config.export_native_model_files = True
    config.env = 'CartPole-v1'
    alg = config.build()
    for _ in range(num_steps):
        alg.train()
    alg.export_policy_checkpoint(ckpt_dir)
    alg.export_policy_model(model_dir)

def restore_saved_model(export_dir):
    if False:
        while True:
            i = 10
    signature_key = tf1.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    g = tf1.Graph()
    with g.as_default():
        with tf1.Session(graph=g) as sess:
            meta_graph_def = tf1.saved_model.load(sess, [tf1.saved_model.tag_constants.SERVING], export_dir)
            print('Model restored!')
            print('Signature Def Information:')
            print(meta_graph_def.signature_def[signature_key])
            print('You can inspect the model using TensorFlow SavedModel CLI.')
            print('https://www.tensorflow.org/guide/saved_model')

def restore_policy_from_checkpoint(export_dir):
    if False:
        for i in range(10):
            print('nop')
    policy = Policy.from_checkpoint(export_dir)
    test_obs = np.array([0.1, 0.2, 0.3, 0.4])
    results = policy.compute_single_action(test_obs)
    assert len(results) == 3
    assert results[0].shape == ()
    assert results[1] == []
    assert results[2]['action_dist_inputs'].shape == (2,)
if __name__ == '__main__':
    algo = 'PPO'
    model_dir = os.path.join(ray._private.utils.get_user_temp_dir(), 'model_export_dir')
    ckpt_dir = os.path.join(ray._private.utils.get_user_temp_dir(), 'ckpt_export_dir')
    num_steps = 1
    train_and_export_policy_and_model(algo, num_steps, model_dir, ckpt_dir)
    restore_saved_model(model_dir)
    restore_policy_from_checkpoint(ckpt_dir)