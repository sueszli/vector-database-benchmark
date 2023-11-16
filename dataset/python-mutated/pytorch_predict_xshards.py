from process_xshards import get_feature_cols
from pytorch_model import NCF
from utils import *
from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca.data import XShards
args = parse_args('PyTorch NCF Prediction with Orca XShards', mode='predict')
init_orca(args.cluster_mode, extra_python_lib='process_xshards.py,pytorch_model.py,utils.py')
data = XShards.load_pickle(os.path.join(args.data_dir, 'test_processed_xshards'))

def model_creator(config):
    if False:
        return 10
    model = NCF(user_num=config['user_num'], item_num=config['item_num'], factor_num=config['factor_num'], num_layers=config['num_layers'], dropout=config['dropout'], model='NeuMF-end', sparse_feats_input_dims=config['sparse_feats_input_dims'], sparse_feats_embed_dims=config['sparse_feats_embed_dims'], num_dense_feats=config['num_dense_feats'])
    return model
config = load_model_config(args.model_dir, 'config.json')
est = Estimator.from_torch(model=model_creator, config=config, backend=args.backend, workers_per_node=args.workers_per_node)
est.load(os.path.join(args.model_dir, 'NCF_model'))
predictions = est.predict(data, feature_cols=get_feature_cols(), batch_size=10240)
print('Prediction results of the first 5 rows:')
print(predictions.head(n=5))
predictions.save_pickle(os.path.join(args.data_dir, 'test_predictions_xshards'))
est.shutdown()
stop_orca_context()