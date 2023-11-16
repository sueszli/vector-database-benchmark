import multiprocessing
import os
import random
import shutil
import sys
import traceback
import unittest
import uuid
from distutils.util import strtobool
import cloudpickle
import numpy as np
import pandas as pd
from ludwig.api import LudwigModel
from ludwig.backend import LocalBackend
from ludwig.constants import VECTOR, COLUMN, NAME, PROC_COLUMN
from ludwig.data.dataset_synthesizer import DATETIME_FORMATS
from ludwig.data.dataset_synthesizer import build_synthetic_dataset
from ludwig.experiment import experiment_cli
from ludwig.features.feature_utils import compute_feature_hash
from ludwig.utils.data_utils import read_csv, replace_file_extension
ENCODERS = ['embed', 'rnn', 'parallel_cnn', 'cnnrnn', 'stacked_parallel_cnn', 'stacked_cnn', 'transformer']
HF_ENCODERS_SHORT = ['distilbert']
HF_ENCODERS = ['bert', 'gpt', 'gpt2', 'xlnet', 'xlm', 'roberta', 'distilbert', 'ctrl', 'camembert', 'albert', 't5', 'xlmroberta', 'longformer', 'flaubert', 'electra', 'mt5']

class LocalTestBackend(LocalBackend):

    @property
    def supports_multiprocessing(self):
        if False:
            while True:
                i = 10
        return False

def parse_flag_from_env(key, default=False):
    if False:
        i = 10
        return i + 15
    try:
        value = os.environ[key]
    except KeyError:
        _value = default
    else:
        try:
            _value = strtobool(value)
        except ValueError:
            raise ValueError('If set, {} must be yes or no.'.format(key))
    return _value
_run_slow_tests = parse_flag_from_env('RUN_SLOW', default=False)

def slow(test_case):
    if False:
        print('Hello World!')
    '\n    Decorator marking a test as slow.\n\n    Slow tests are skipped by default. Set the RUN_SLOW environment variable\n    to a truth value to run them.\n\n    '
    if not _run_slow_tests:
        test_case = unittest.skip('Skipping: this test is too slow')(test_case)
    return test_case

def generate_data(input_features, output_features, filename='test_csv.csv', num_examples=25):
    if False:
        for i in range(10):
            print('nop')
    '\n    Helper method to generate synthetic data based on input, output feature\n    specs\n    :param num_examples: number of examples to generate\n    :param input_features: schema\n    :param output_features: schema\n    :param filename: path to the file where data is stored\n    :return:\n    '
    features = input_features + output_features
    df = build_synthetic_dataset(num_examples, features)
    data = [next(df) for _ in range(num_examples)]
    dataframe = pd.DataFrame(data[1:], columns=data[0])
    dataframe.to_csv(filename, index=False)
    return filename

def random_string(length=5):
    if False:
        print('Hello World!')
    return uuid.uuid4().hex[:length].upper()

def numerical_feature(normalization=None, **kwargs):
    if False:
        return 10
    feature = {'name': 'num_' + random_string(), 'type': 'number', 'preprocessing': {'normalization': normalization}}
    feature.update(kwargs)
    feature[COLUMN] = feature[NAME]
    feature[PROC_COLUMN] = compute_feature_hash(feature)
    return feature

def category_feature(**kwargs):
    if False:
        while True:
            i = 10
    feature = {'type': 'category', 'name': 'category_' + random_string(), 'vocab_size': 10, 'embedding_size': 5}
    feature.update(kwargs)
    feature[COLUMN] = feature[NAME]
    feature[PROC_COLUMN] = compute_feature_hash(feature)
    return feature

def text_feature(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    feature = {'name': 'text_' + random_string(), 'type': 'text', 'reduce_input': None, 'vocab_size': 5, 'min_len': 7, 'max_len': 7, 'embedding_size': 8, 'state_size': 8}
    feature.update(kwargs)
    feature[COLUMN] = feature[NAME]
    feature[PROC_COLUMN] = compute_feature_hash(feature)
    return feature

def set_feature(**kwargs):
    if False:
        return 10
    feature = {'type': 'set', 'name': 'set_' + random_string(), 'vocab_size': 10, 'max_len': 5, 'embedding_size': 5}
    feature.update(kwargs)
    feature[COLUMN] = feature[NAME]
    feature[PROC_COLUMN] = compute_feature_hash(feature)
    return feature

def sequence_feature(**kwargs):
    if False:
        return 10
    feature = {'type': 'sequence', 'name': 'sequence_' + random_string(), 'vocab_size': 10, 'max_len': 7, 'encoder': 'embed', 'embedding_size': 8, 'fc_size': 8, 'state_size': 8, 'num_filters': 8, 'hidden_size': 8}
    feature.update(kwargs)
    feature[COLUMN] = feature[NAME]
    feature[PROC_COLUMN] = compute_feature_hash(feature)
    return feature

def image_feature(folder, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    feature = {'type': 'image', 'name': 'image_' + random_string(), 'encoder': 'resnet', 'preprocessing': {'in_memory': True, 'height': 12, 'width': 12, 'num_channels': 3}, 'resnet_size': 8, 'destination_folder': folder, 'fc_size': 8, 'num_filters': 8}
    feature.update(kwargs)
    feature[COLUMN] = feature[NAME]
    feature[PROC_COLUMN] = compute_feature_hash(feature)
    return feature

def audio_feature(folder, **kwargs):
    if False:
        return 10
    feature = {'name': 'audio_' + random_string(), 'type': 'audio', 'preprocessing': {'audio_feature': {'type': 'fbank', 'window_length_in_s': 0.04, 'window_shift_in_s': 0.02, 'num_filter_bands': 80}, 'audio_file_length_limit_in_s': 3.0}, 'encoder': 'stacked_cnn', 'should_embed': False, 'conv_layers': [{'filter_size': 400, 'pool_size': 16, 'num_filters': 32, 'regularize': 'false'}, {'filter_size': 40, 'pool_size': 10, 'num_filters': 64, 'regularize': 'false'}], 'fc_size': 256, 'destination_folder': folder}
    feature.update(kwargs)
    feature[COLUMN] = feature[NAME]
    feature[PROC_COLUMN] = compute_feature_hash(feature)
    return feature

def timeseries_feature(**kwargs):
    if False:
        i = 10
        return i + 15
    feature = {'name': 'timeseries_' + random_string(), 'type': 'timeseries', 'max_len': 7}
    feature.update(kwargs)
    feature[COLUMN] = feature[NAME]
    feature[PROC_COLUMN] = compute_feature_hash(feature)
    return feature

def binary_feature(**kwargs):
    if False:
        print('Hello World!')
    feature = {'name': 'binary_' + random_string(), 'type': 'binary'}
    feature.update(kwargs)
    feature[COLUMN] = feature[NAME]
    feature[PROC_COLUMN] = compute_feature_hash(feature)
    return feature

def bag_feature(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    feature = {'name': 'bag_' + random_string(), 'type': 'bag', 'max_len': 5, 'vocab_size': 10, 'embedding_size': 5}
    feature.update(kwargs)
    feature[COLUMN] = feature[NAME]
    feature[PROC_COLUMN] = compute_feature_hash(feature)
    return feature

def date_feature(**kwargs):
    if False:
        while True:
            i = 10
    feature = {'name': 'date_' + random_string(), 'type': 'date', 'preprocessing': {'datetime_format': random.choice(list(DATETIME_FORMATS.keys()))}}
    feature.update(kwargs)
    feature[COLUMN] = feature[NAME]
    feature[PROC_COLUMN] = compute_feature_hash(feature)
    return feature

def h3_feature(**kwargs):
    if False:
        print('Hello World!')
    feature = {'name': 'h3_' + random_string(), 'type': 'h3'}
    feature.update(kwargs)
    feature[COLUMN] = feature[NAME]
    feature[PROC_COLUMN] = compute_feature_hash(feature)
    return feature

def vector_feature(**kwargs):
    if False:
        return 10
    feature = {'type': VECTOR, 'vector_size': 5, 'name': 'vector_' + random_string()}
    feature.update(kwargs)
    feature[COLUMN] = feature[NAME]
    feature[PROC_COLUMN] = compute_feature_hash(feature)
    return feature

def run_experiment(input_features, output_features, skip_save_processed_input=True, config=None, backend=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Helper method to avoid code repetition in running an experiment. Deletes\n    the data saved to disk after running the experiment\n    :param input_features: list of input feature dictionaries\n    :param output_features: list of output feature dictionaries\n    **kwargs you may also pass extra parameters to the experiment as keyword\n    arguments\n    :return: None\n    '
    if input_features is not None and output_features is not None:
        config = {'input_features': input_features, 'output_features': output_features, 'combiner': {'type': 'concat', 'fc_size': 14}, 'training': {'epochs': 2}}
    args = {'config': config, 'backend': backend or LocalTestBackend(), 'skip_save_training_description': True, 'skip_save_training_statistics': True, 'skip_save_processed_input': skip_save_processed_input, 'skip_save_progress': True, 'skip_save_unprocessed_output': True, 'skip_save_model': True, 'skip_save_predictions': True, 'skip_save_eval_stats': True, 'skip_collect_predictions': True, 'skip_collect_overall_stats': True, 'skip_save_log': True}
    args.update(kwargs)
    (_, _, _, _, exp_dir_name) = experiment_cli(**args)
    shutil.rmtree(exp_dir_name, ignore_errors=True)

def generate_output_features_with_dependencies(main_feature, dependencies):
    if False:
        for i in range(10):
            print('nop')
    output_features = [category_feature(vocab_size=2, reduce_input='sum'), sequence_feature(vocab_size=10, max_len=5), numerical_feature()]
    feature_names = {'feat1': (0, output_features[0]['name']), 'feat2': (1, output_features[1]['name']), 'feat3': (2, output_features[2]['name'])}
    generated_dependencies = [feature_names[feat_name][1] for feat_name in dependencies]
    output_features[feature_names[main_feature][0]]['dependencies'] = generated_dependencies
    return output_features

def _subproc_wrapper(fn, queue, *args, **kwargs):
    if False:
        while True:
            i = 10
    fn = cloudpickle.loads(fn)
    try:
        results = fn(*args, **kwargs)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        results = e
    queue.put(results)

def spawn(fn):
    if False:
        return 10

    def wrapped_fn(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        ctx = multiprocessing.get_context('spawn')
        queue = ctx.Queue()
        p = ctx.Process(target=_subproc_wrapper, args=(cloudpickle.dumps(fn), queue, *args), kwargs=kwargs)
        p.start()
        p.join()
        results = queue.get()
        if isinstance(results, Exception):
            raise RuntimeError(f'Spawned subprocess raised {type(results).__name__}, check log output above for stack trace.')
        return results
    return wrapped_fn

def run_api_experiment(input_features, output_features, data_csv):
    if False:
        for i in range(10):
            print('nop')
    '\n    Helper method to avoid code repetition in running an experiment\n    :param input_features: input schema\n    :param output_features: output schema\n    :param data_csv: path to data\n    :return: None\n    '
    config = {'input_features': input_features, 'output_features': output_features, 'combiner': {'type': 'concat', 'fc_size': 14}, 'training': {'epochs': 2}}
    model = LudwigModel(config)
    output_dir = None
    try:
        (_, _, output_dir) = model.train(dataset=data_csv, skip_save_processed_input=True, skip_save_progress=True, skip_save_unprocessed_output=True)
        model.predict(dataset=data_csv)
        model_dir = os.path.join(output_dir, 'model')
        loaded_model = LudwigModel.load(model_dir)
        loaded_model.predict(dataset=data_csv)
        model_weights = model.model.get_weights()
        loaded_weights = loaded_model.model.get_weights()
        for (model_weight, loaded_weight) in zip(model_weights, loaded_weights):
            assert np.allclose(model_weight, loaded_weight)
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
    try:
        data_df = read_csv(data_csv)
        (_, _, output_dir) = model.train(dataset=data_df, skip_save_processed_input=True, skip_save_progress=True, skip_save_unprocessed_output=True)
        model.predict(dataset=data_df)
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)

def create_data_set_to_use(data_format, raw_data):
    if False:
        i = 10
        return i + 15
    from ray._private.thirdparty.tabulate.tabulate import tabulate

    def to_fwf(df, fname):
        if False:
            return 10
        content = tabulate(df.values.tolist(), list(df.columns), tablefmt='plain')
        open(fname, 'w').write(content)
    pd.DataFrame.to_fwf = to_fwf
    dataset_to_use = None
    if data_format == 'csv':
        dataset_to_use = raw_data
    elif data_format in {'df', 'dict'}:
        dataset_to_use = pd.read_csv(raw_data)
        if data_format == 'dict':
            dataset_to_use = dataset_to_use.to_dict(orient='list')
    elif data_format == 'excel':
        dataset_to_use = replace_file_extension(raw_data, 'xlsx')
        pd.read_csv(raw_data).to_excel(dataset_to_use, index=False)
    elif data_format == 'excel_xls':
        dataset_to_use = replace_file_extension(raw_data, 'xls')
        pd.read_csv(raw_data).to_excel(dataset_to_use, index=False)
    elif data_format == 'feather':
        dataset_to_use = replace_file_extension(raw_data, 'feather')
        pd.read_csv(raw_data).to_feather(dataset_to_use)
    elif data_format == 'fwf':
        dataset_to_use = replace_file_extension(raw_data, 'fwf')
        pd.read_csv(raw_data).to_fwf(dataset_to_use)
    elif data_format == 'html':
        dataset_to_use = replace_file_extension(raw_data, 'html')
        pd.read_csv(raw_data).to_html(dataset_to_use, index=False)
    elif data_format == 'json':
        dataset_to_use = replace_file_extension(raw_data, 'json')
        pd.read_csv(raw_data).to_json(dataset_to_use, orient='records')
    elif data_format == 'jsonl':
        dataset_to_use = replace_file_extension(raw_data, 'jsonl')
        pd.read_csv(raw_data).to_json(dataset_to_use, orient='records', lines=True)
    elif data_format == 'parquet':
        dataset_to_use = replace_file_extension(raw_data, 'parquet')
        pd.read_csv(raw_data).to_parquet(dataset_to_use, index=False)
    elif data_format == 'pickle':
        dataset_to_use = replace_file_extension(raw_data, 'pickle')
        pd.read_csv(raw_data).to_pickle(dataset_to_use)
    elif data_format == 'stata':
        dataset_to_use = replace_file_extension(raw_data, 'stata')
        pd.read_csv(raw_data).to_stata(dataset_to_use)
    elif data_format == 'tsv':
        dataset_to_use = replace_file_extension(raw_data, 'tsv')
        pd.read_csv(raw_data).to_csv(dataset_to_use, sep='\t', index=False)
    else:
        ValueError("'{}' is an unrecognized data format".format(data_format))
    return dataset_to_use

def train_with_backend(backend, config, dataset=None, training_set=None, validation_set=None, test_set=None, predict=True, evaluate=True):
    if False:
        i = 10
        return i + 15
    model = LudwigModel(config, backend=backend)
    output_dir = None
    ret = False
    try:
        (_, _, output_dir) = model.train(dataset=dataset, training_set=training_set, validation_set=validation_set, test_set=test_set, skip_save_processed_input=True, skip_save_progress=True, skip_save_unprocessed_output=True)
        if dataset is None:
            dataset = training_set
        if predict:
            (preds, _) = model.predict(dataset=dataset)
            assert backend.df_engine.compute(preds) is not None
        if evaluate:
            (_, eval_preds, _) = model.evaluate(dataset=dataset)
            assert backend.df_engine.compute(eval_preds) is not None
        ret = True
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
    return ret