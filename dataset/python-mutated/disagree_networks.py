"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf
"""
from ray.rllib.algorithms.dreamerv3.tf.models.components.mlp import MLP
from ray.rllib.algorithms.dreamerv3.tf.models.components.representation_layer import RepresentationLayer
from ray.rllib.utils.framework import try_import_tf, try_import_tfp
(_, tf, _) = try_import_tf()
tfp = try_import_tfp()

class DisagreeNetworks(tf.keras.Model):
    """Predict the RSSM's z^(t+1), given h(t), z^(t), and a(t).

    Disagreement (stddev) between the N networks in this model on what the next z^ would
    be are used to produce intrinsic rewards for enhanced, curiosity-based exploration.

    TODO
    """

    def __init__(self, *, num_networks, model_size, intrinsic_rewards_scale):
        if False:
            print('Hello World!')
        super().__init__(name='disagree_networks')
        self.model_size = model_size
        self.num_networks = num_networks
        self.intrinsic_rewards_scale = intrinsic_rewards_scale
        self.mlps = []
        self.representation_layers = []
        for _ in range(self.num_networks):
            self.mlps.append(MLP(model_size=self.model_size, output_layer_size=None, trainable=True))
            self.representation_layers.append(RepresentationLayer(model_size=self.model_size, name='disagree'))

    def call(self, inputs, z, a, training=None):
        if False:
            while True:
                i = 10
        return self.forward_train(a=a, h=inputs, z=z)

    def compute_intrinsic_rewards(self, h, z, a):
        if False:
            print('Hello World!')
        forward_train_outs = self.forward_train(a=a, h=h, z=z)
        B = tf.shape(h)[0]
        z_predicted_probs_N_B = forward_train_outs['z_predicted_probs_N_HxB']
        N = len(z_predicted_probs_N_B)
        z_predicted_probs_N_B = tf.stack(z_predicted_probs_N_B, axis=0)
        z_predicted_probs_N_B = tf.reshape(z_predicted_probs_N_B, shape=(N, B, -1))
        stddevs_B_mean = tf.reduce_mean(tf.math.reduce_std(z_predicted_probs_N_B, axis=0), axis=-1)
        stddevs_B_mean -= tf.reduce_mean(stddevs_B_mean)
        return {'rewards_intrinsic': stddevs_B_mean * self.intrinsic_rewards_scale, 'forward_train_outs': forward_train_outs}

    def forward_train(self, a, h, z):
        if False:
            return 10
        HxB = tf.shape(h)[0]
        z = tf.reshape(z, shape=(HxB, -1))
        inputs_ = tf.stop_gradient(tf.concat([h, z, a], axis=-1))
        z_predicted_probs_N_HxB = [repr(mlp(inputs_))[1] for (mlp, repr) in zip(self.mlps, self.representation_layers)]
        return {'z_predicted_probs_N_HxB': z_predicted_probs_N_HxB}