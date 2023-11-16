import math
import tensorflow as tf
from process_xshards import prepare_data
from tf_model import ncf_model
from utils import *
from bigdl.orca.learn.tf2 import Estimator
args = parse_args('TensorFlow NCF Training with Orca XShards')
init_orca(args.cluster_mode, extra_python_lib='process_xshards.py,tf_model.py,utils.py')
(train_data, test_data, user_num, item_num, sparse_feats_input_dims, num_dense_feats, feature_cols, label_cols) = prepare_data(args.data_dir, args.dataset, num_ng=4)
config = dict(user_num=user_num, item_num=item_num, factor_num=16, num_layers=3, dropout=0.5, lr=0.01, sparse_feats_input_dims=sparse_feats_input_dims, sparse_feats_embed_dims=8, num_dense_feats=num_dense_feats)

def model_creator(config):
    if False:
        return 10
    model = ncf_model(user_num=config['user_num'], item_num=config['item_num'], num_layers=config['num_layers'], factor_num=config['factor_num'], dropout=config['dropout'], lr=config['lr'], sparse_feats_input_dims=config['sparse_feats_input_dims'], sparse_feats_embed_dims=config['sparse_feats_embed_dims'], num_dense_feats=config['num_dense_feats'])
    return model
est = Estimator.from_keras(model_creator=model_creator, config=config, backend=args.backend, workers_per_node=args.workers_per_node)
batch_size = 10240
train_steps = math.ceil(len(train_data) / batch_size)
val_steps = math.ceil(len(test_data) / batch_size)
callbacks = [tf.keras.callbacks.TensorBoard(log_dir=os.path.join(args.model_dir, 'logs'))] if args.tensorboard else []
if args.lr_scheduler:
    lr_callback = tf.keras.callbacks.LearningRateScheduler(schedule_func, verbose=1)
    callbacks.append(lr_callback)
train_stats = est.fit(train_data, epochs=2, batch_size=batch_size, feature_cols=feature_cols, label_cols=label_cols, steps_per_epoch=train_steps, validation_data=test_data, validation_steps=val_steps, callbacks=callbacks)
print('Train results:')
for (k, v) in train_stats.items():
    print('{}: {}'.format(k, v))
eval_stats = est.evaluate(test_data, feature_cols=feature_cols, label_cols=label_cols, batch_size=batch_size, num_steps=val_steps)
print('Evaluation results:')
for (k, v) in eval_stats.items():
    print('{}: {}'.format(k, v))
est.save(os.path.join(args.model_dir, 'NCF_model.h5'))
save_model_config(config, args.model_dir, 'config.json')
train_data.save_pickle(os.path.join(args.data_dir, 'train_processed_xshards'))
test_data.save_pickle(os.path.join(args.data_dir, 'test_processed_xshards'))
est.shutdown()
stop_orca_context()