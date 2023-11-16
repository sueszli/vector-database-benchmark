"""SVTCN estimator implementation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import data_providers
import model as model_module
from estimators import base_estimator
from estimators import svtcn_loss
import tensorflow as tf

class SVTCNEstimator(base_estimator.BaseEstimator):
    """Single-view TCN Estimator base class."""

    def __init__(self, config, logdir):
        if False:
            while True:
                i = 10
        super(SVTCNEstimator, self).__init__(config, logdir)

    def construct_input_fn(self, records, is_training):
        if False:
            return 10
        'See base class.'
        config = self._config
        num_views = config.data.num_views
        num_parallel_calls = config.data.num_parallel_calls
        sequence_prefetch_size = config.data.sequence_prefetch_size
        batch_prefetch_size = config.data.batch_prefetch_size

        def input_fn():
            if False:
                print('Hello World!')
            'Provides input to SVTCN models.'
            (images_preprocessed, images_raw, timesteps) = data_providers.singleview_tcn_provider(file_list=records, preprocess_fn=self.preprocess_data, num_views=num_views, is_training=is_training, batch_size=self._batch_size, num_parallel_calls=num_parallel_calls, sequence_prefetch_size=sequence_prefetch_size, batch_prefetch_size=batch_prefetch_size)
            if config.logging.summary.image_summaries and is_training:
                tf.summary.image('training/svtcn_images', images_raw)
            features = {'batch_preprocessed': images_preprocessed}
            return (features, timesteps)
        return input_fn

    def forward(self, images, is_training, reuse=False):
        if False:
            return 10
        'See base class.'
        embedder_strategy = self._config.embedder_strategy
        embedder = model_module.get_embedder(embedder_strategy, self._config, images, is_training=is_training, reuse=reuse)
        embeddings = embedder.construct_embedding()
        if is_training:
            self.variables_to_train = embedder.get_trainable_variables()
            self.pretrained_init_fn = embedder.init_fn
        return embeddings

class SVTCNTripletEstimator(SVTCNEstimator):
    """Single-View TCN with semihard triplet loss."""

    def __init__(self, config, logdir):
        if False:
            return 10
        super(SVTCNTripletEstimator, self).__init__(config, logdir)

    def define_loss(self, embeddings, timesteps, is_training):
        if False:
            while True:
                i = 10
        'See base class.'
        pos_radius = self._config.svtcn.pos_radius
        neg_radius = self._config.svtcn.neg_radius
        margin = self._config.triplet_semihard.margin
        loss = svtcn_loss.singleview_tcn_loss(embeddings, timesteps, pos_radius, neg_radius, margin=margin)
        self._loss = loss
        if is_training:
            tf.summary.scalar('training/svtcn_loss', loss)
        return loss

    def define_eval_metric_ops(self):
        if False:
            i = 10
            return i + 15
        'See base class.'
        return {'validation/svtcn_loss': tf.metrics.mean(self._loss)}