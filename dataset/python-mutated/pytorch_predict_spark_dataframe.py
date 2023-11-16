from process_spark_dataframe import get_feature_cols
from pytorch_model import NCF
from utils import *
from bigdl.orca.learn.pytorch import Estimator
args = parse_args('PyTorch NCF Prediction with Spark DataFrame', mode='predict')
init_orca(args.cluster_mode, extra_python_lib='process_spark_dataframe.py,pytorch_model.py,utils.py')
spark = OrcaContext.get_spark_session()
df = spark.read.parquet(os.path.join(args.data_dir, 'test_processed_dataframe.parquet'))

def model_creator(config):
    if False:
        return 10
    model = NCF(user_num=config['user_num'], item_num=config['item_num'], factor_num=config['factor_num'], num_layers=config['num_layers'], dropout=config['dropout'], model='NeuMF-end', sparse_feats_input_dims=config['sparse_feats_input_dims'], sparse_feats_embed_dims=config['sparse_feats_embed_dims'], num_dense_feats=config['num_dense_feats'])
    return model
config = load_model_config(args.model_dir, 'config.json')
est = Estimator.from_torch(model=model_creator, config=config, backend=args.backend, workers_per_node=args.workers_per_node)
est.load(os.path.join(args.model_dir, 'NCF_model'))
predict_df = est.predict(df, feature_cols=get_feature_cols(), batch_size=10240)
print('Prediction results of the first 5 rows:')
predict_df.show(5)
predict_df.write.parquet(os.path.join(args.data_dir, 'test_predictions_dataframe.parquet'), mode='overwrite')
est.shutdown()
stop_orca_context()