import sys
sys.path.append('/home/yooyeonsung/fever-new/allennlp-0.2.3/')
sys.path.append('/home/yooyeonsung/fever-new/allennlp-0.2.3/allennlp')
import os
import pdb
import pickle
from copy import deepcopy
from allennlp.commands.train import prepare_environment
from typing import List, Union, Dict, Any
from allennlp.common import Params
from allennlp.common.tee_logger import TeeLogger
from allennlp.data import Vocabulary, Dataset, DataIterator, DatasetReader, Tokenizer, TokenIndexer
from allennlp.models import Model, archive_model
from allennlp.training import Trainer
from common.util.log_helper import LogHelper
from common.util.random import SimpleRandom
from retrieval.fever_doc_db import FeverDocDB
from rte.parikh.reader import FEVERReader
import argparse
import logging
import sys
import json
logger = logging.getLogger(__name__)

def train_model(db: FeverDocDB, params: Union[Params, Dict[str, Any]], cuda_device: int, serialization_dir: str, filtering: str) -> Model:
    if False:
        return 10
    '\n    This function can be used as an entry point to running models in AllenNLP\n    directly from a JSON specification using a :class:`Driver`. Note that if\n    you care about reproducibility, you should avoid running code using Pytorch\n    or numpy which affect the reproducibility of your experiment before you\n    import and use this function, these libraries rely on random seeds which\n    can be set in this function via a JSON specification file. Note that this\n    function performs training and will also evaluate the trained model on\n    development and test sets if provided in the parameter json.\n\n    Parameters\n    ----------\n    params: Params, required.\n        A parameter object specifying an AllenNLP Experiment.\n    serialization_dir: str, required\n        The directory in which to save results and logs.\n    '
    SimpleRandom.set_seeds()
    os.makedirs(serialization_dir, exist_ok=True)
    sys.stdout = TeeLogger(os.path.join(serialization_dir, 'stdout.log'), sys.stdout)
    sys.stderr = TeeLogger(os.path.join(serialization_dir, 'stderr.log'), sys.stderr)
    handler = logging.FileHandler(os.path.join(serialization_dir, 'python_logging.log'))
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
    logging.getLogger().addHandler(handler)
    serialization_params = deepcopy(params).as_dict(quiet=True)
    with open(os.path.join(serialization_dir, 'model_params.json'), 'w') as param_file:
        json.dump(serialization_params, param_file, indent=4)
    ds_params = params.pop('dataset_reader', {})
    dataset_reader = FEVERReader(db, sentence_level=ds_params.pop('sentence_level', False), wiki_tokenizer=Tokenizer.from_params(ds_params.pop('wiki_tokenizer', {})), claim_tokenizer=Tokenizer.from_params(ds_params.pop('claim_tokenizer', {})), token_indexers=TokenIndexer.dict_from_params(ds_params.pop('token_indexers', {})), filtering=filtering)
    train_data_path = params.pop('train_data_path')
    logger.info('Reading training data from %s', train_data_path)
    train_data = dataset_reader.read(train_data_path)
    all_datasets = [train_data]
    datasets_in_vocab = ['train']
    validation_data_path = params.pop('validation_data_path', None)
    if validation_data_path is not None:
        logger.info('Reading validation data from %s', validation_data_path)
        validation_data = dataset_reader.read(validation_data_path)
        all_datasets.append(validation_data)
        datasets_in_vocab.append('validation')
    else:
        validation_data = None
    logger.info('Creating a vocabulary using %s data.', ', '.join(datasets_in_vocab))
    vocab = Vocabulary.from_params(params.pop('vocabulary', {}), Dataset([instance for dataset in all_datasets for instance in dataset.instances]))
    pdb.set_trace()
    vocab.save_to_files(os.path.join(serialization_dir, 'vocabulary'))
    model = Model.from_params(vocab, params.pop('model'))
    iterator = DataIterator.from_params(params.pop('iterator'))
    train_data.index_instances(vocab)
    if validation_data:
        validation_data.index_instances(vocab)
    trainer_params = params.pop('trainer')
    if cuda_device is not None:
        trainer_params['cuda_device'] = cuda_device
    trainer = Trainer.from_params(model, serialization_dir, iterator, train_data, validation_data, trainer_params)
    trainer.train()
    archive_model(serialization_dir)
    return model
if __name__ == '__main__':
    LogHelper.setup()
    LogHelper.get_logger('allennlp.training.trainer')
    LogHelper.get_logger(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument('db', type=str, help='/path/to/saved/db.db')
    parser.add_argument('param_path', type=str, help='path to parameter file describing the model to be trained')
    parser.add_argument('logdir', type=str)
    parser.add_argument('--filtering', type=str, default=None)
    parser.add_argument('--cuda-device', type=int, default=None, help='id of GPU to use (if any)')
    parser.add_argument('-o', '--overrides', type=str, default='', help='a HOCON structure used to override the experiment configuration')
    args = parser.parse_args()
    db = FeverDocDB(args.db)
    params = Params.from_file(args.param_path, args.overrides)
    train_model(db, params, args.cuda_device, args.logdir, args.filtering)