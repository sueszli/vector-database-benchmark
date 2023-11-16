import torch.nn as nn
import torch.optim as optim
from process_xshards import prepare_data
from pytorch_model import NCF
from utils import *
from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca.learn.metrics import Accuracy, Precision, Recall
args = parse_args('PyTorch NCF Training with Orca XShards')
init_orca(args.cluster_mode, extra_python_lib='process_xshards.py,pytorch_model.py,utils.py')
(train_data, test_data, user_num, item_num, sparse_feats_input_dims, num_dense_feats, feature_cols, label_cols) = prepare_data(args.data_dir, args.dataset, num_ng=4)
config = dict(user_num=user_num, item_num=item_num, factor_num=16, num_layers=3, dropout=0.5, lr=0.01, sparse_feats_input_dims=sparse_feats_input_dims, sparse_feats_embed_dims=8, num_dense_feats=num_dense_feats)

def model_creator(config):
    if False:
        for i in range(10):
            print('nop')
    model = NCF(user_num=config['user_num'], item_num=config['item_num'], factor_num=config['factor_num'], num_layers=config['num_layers'], dropout=config['dropout'], model='NeuMF-end', sparse_feats_input_dims=config['sparse_feats_input_dims'], sparse_feats_embed_dims=config['sparse_feats_embed_dims'], num_dense_feats=config['num_dense_feats'])
    model.train()
    return model

def optimizer_creator(model, config):
    if False:
        for i in range(10):
            print('nop')
    return optim.Adam(model.parameters(), lr=config['lr'])

def scheduler_creator(optimizer, config):
    if False:
        for i in range(10):
            print('nop')
    return optim.lr_scheduler.StepLR(optimizer, step_size=1)
loss = nn.BCEWithLogitsLoss()
callbacks = get_pytorch_callbacks(args)
scheduler_creator = scheduler_creator if args.lr_scheduler else None
est = Estimator.from_torch(model=model_creator, optimizer=optimizer_creator, loss=loss, scheduler_creator=scheduler_creator, metrics=[Accuracy(), Precision(), Recall()], config=config, backend=args.backend, use_tqdm=True, workers_per_node=args.workers_per_node)
train_stats = est.fit(train_data, epochs=2, batch_size=10240, feature_cols=feature_cols, label_cols=label_cols, validation_data=test_data, callbacks=callbacks)
print('Train results:')
for epoch_stats in train_stats:
    for (k, v) in epoch_stats.items():
        print('{}: {}'.format(k, v))
    print()
eval_stats = est.evaluate(test_data, feature_cols=feature_cols, label_cols=label_cols, batch_size=10240)
print('Evaluation results:')
for (k, v) in eval_stats.items():
    print('{}: {}'.format(k, v))
est.save(os.path.join(args.model_dir, 'NCF_model'))
save_model_config(config, args.model_dir, 'config.json')
train_data.save_pickle(os.path.join(args.data_dir, 'train_processed_xshards'))
test_data.save_pickle(os.path.join(args.data_dir, 'test_processed_xshards'))
est.shutdown()
stop_orca_context()