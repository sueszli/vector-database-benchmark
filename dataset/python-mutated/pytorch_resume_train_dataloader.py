import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from pytorch_dataset import load_dataset, process_users_items, get_input_dims
from pytorch_model import NCF
from utils import *
from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca.learn.metrics import Accuracy, Precision, Recall
args = parse_args('PyTorch NCF Resume Training with DataLoader')
init_orca(args.cluster_mode, extra_python_lib='pytorch_dataset.py,pytorch_model.py,utils.py')

def train_loader_func(config, batch_size):
    if False:
        while True:
            i = 10
    (train_dataset, _) = load_dataset(config['data_dir'], config['dataset'], num_ng=4)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return train_loader

def test_loader_func(config, batch_size):
    if False:
        i = 10
        return i + 15
    (_, test_dataset) = load_dataset(config['data_dir'], config['dataset'], num_ng=4)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return test_loader

def model_creator(config):
    if False:
        for i in range(10):
            print('nop')
    (users, items, user_num, item_num, sparse_features, dense_features, total_cols) = process_users_items(config['data_dir'], config['dataset'])
    (sparse_feats_input_dims, num_dense_feats) = get_input_dims(users, items, sparse_features, dense_features)
    model = NCF(user_num=user_num, item_num=item_num, factor_num=config['factor_num'], num_layers=config['num_layers'], dropout=config['dropout'], model='NeuMF-end', sparse_feats_input_dims=sparse_feats_input_dims, sparse_feats_embed_dims=config['sparse_feats_embed_dims'], num_dense_feats=num_dense_feats)
    model.train()
    return model

def optimizer_creator(model, config):
    if False:
        return 10
    return optim.Adam(model.parameters(), lr=config['lr'])

def scheduler_creator(optimizer, config):
    if False:
        while True:
            i = 10
    return optim.lr_scheduler.StepLR(optimizer, step_size=1)
loss = nn.BCEWithLogitsLoss()
config = load_model_config(args.model_dir, 'config.json')
callbacks = get_pytorch_callbacks(args)
scheduler_creator = scheduler_creator if args.lr_scheduler else None
est = Estimator.from_torch(model=model_creator, optimizer=optimizer_creator, loss=loss, scheduler_creator=scheduler_creator, metrics=[Accuracy(), Precision(), Recall()], config=config, backend=args.backend, use_tqdm=True, workers_per_node=args.workers_per_node)
est.load(os.path.join(args.model_dir, 'NCF_model'))
train_stats = est.fit(train_loader_func, epochs=2, batch_size=10240, validation_data=test_loader_func, callbacks=callbacks)
print('Train results:')
for epoch_stats in train_stats:
    for (k, v) in epoch_stats.items():
        print('{}: {}'.format(k, v))
    print()
est.save(os.path.join(args.model_dir, 'NCF_resume_model'))
est.shutdown()
stop_orca_context()