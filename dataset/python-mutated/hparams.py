"""Define model HParams."""
import tensorflow as tf

def create_hparams(hparam_string=None):
    if False:
        while True:
            i = 10
    'Create model hyperparameters. Parse nondefault from given string.'
    hparams = tf.contrib.training.HParams(arch='resnet', lrelu_leakiness=0.2, batch_norm_decay=0.9, weight_decay=1e-05, normal_init_std=0.02, generator_kernel_size=3, discriminator_kernel_size=3, num_training_examples=0, augment_source_images=False, augment_target_images=False, num_discriminator_filters=64, discriminator_conv_block_size=1, discriminator_filter_factor=2.0, discriminator_noise_stddev=0.2, discriminator_image_noise=False, discriminator_first_stride=1, discriminator_do_pooling=False, discriminator_dropout_keep_prob=0.9, num_decoder_filters=64, num_encoder_filters=64, projection_shape_size=4, projection_shape_channels=64, upsample_method='resize_conv', summary_steps=500, task_tower='doubling_pose_estimator', weight_decay_task_classifier=1e-05, source_task_loss_weight=1.0, transferred_task_loss_weight=1.0, num_private_layers=2, source_pose_weight=0.125 * 2.0, transferred_pose_weight=0.125 * 1.0, task_tower_in_g_step=True, task_loss_in_g_weight=1.0, simple_num_conv_layers=1, simple_conv_filters=8, resnet_blocks=6, resnet_filters=64, resnet_residuals=True, res_int_blocks=2, res_int_convs=2, res_int_filters=64, noise_channel=True, noise_dims=10, condition_on_source_class=False, domain_loss_weight=1.0, style_transfer_loss_weight=1.0, transferred_similarity_loss_weight=0.0, transferred_similarity_loss='mpse', transferred_similarity_max_diff=0.4, learning_rate=0.001, batch_size=32, lr_decay_steps=20000, lr_decay_rate=0.95, adam_beta1=0.5, clip_gradient_norm=5.0, discriminator_steps=1, generator_steps=1)
    if hparam_string:
        tf.logging.info('Parsing command line hparams: %s', hparam_string)
        hparams.parse(hparam_string)
    tf.logging.info('Final parsed hparams: %s', hparams.values())
    return hparams