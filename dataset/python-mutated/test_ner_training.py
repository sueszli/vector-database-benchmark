import json
import logging
import os
import warnings
import pytest
import torch
pytestmark = [pytest.mark.travis, pytest.mark.pipeline]
from stanza.models import ner_tagger
from stanza.models.ner.trainer import Trainer
from stanza.tests import TEST_WORKING_DIR
from stanza.utils.datasets.ner.prepare_ner_file import process_dataset
logger = logging.getLogger('stanza')
EN_TRAIN_BIO = '\nChris B-PERSON\nManning E-PERSON\nis O\na O\ngood O\nman O\n. O\n\nHe O\nworks O\nin O\nStanford B-ORG\nUniversity E-ORG\n. O\n'.lstrip().replace(' ', '\t')
EN_DEV_BIO = '\nChris B-PERSON\nManning E-PERSON\nis O\npart O\nof O\nComputer B-ORG\nScience E-ORG\n'.lstrip().replace(' ', '\t')
EN_TRAIN_2TAG = '\nChris B-PERSON B-PER\nManning E-PERSON E-PER\nis O O\na O O\ngood O O\nman O O\n. O O\n\nHe O O\nworks O O\nin O O\nStanford B-ORG B-ORG\nUniversity E-ORG B-ORG\n. O O\n'.strip().replace(' ', '\t')
EN_TRAIN_2TAG_EMPTY2 = '\nChris B-PERSON -\nManning E-PERSON -\nis O -\na O -\ngood O -\nman O -\n. O -\n\nHe O -\nworks O -\nin O -\nStanford B-ORG -\nUniversity E-ORG -\n. O -\n'.strip().replace(' ', '\t')
EN_DEV_2TAG = '\nChris B-PERSON B-PER\nManning E-PERSON E-PER\nis O O\npart O O\nof O O\nComputer B-ORG B-ORG\nScience E-ORG E-ORG\n'.strip().replace(' ', '\t')

@pytest.fixture(scope='module')
def pretrain_file():
    if False:
        while True:
            i = 10
    return f'{TEST_WORKING_DIR}/in/tiny_emb.pt'

def write_temp_file(filename, bio_data):
    if False:
        for i in range(10):
            print('nop')
    bio_filename = os.path.splitext(filename)[0] + '.bio'
    with open(bio_filename, 'w', encoding='utf-8') as fout:
        fout.write(bio_data)
    process_dataset(bio_filename, filename)

def write_temp_2tag(filename, bio_data):
    if False:
        i = 10
        return i + 15
    doc = []
    sentences = bio_data.split('\n\n')
    for sentence in sentences:
        doc.append([])
        for word in sentence.split('\n'):
            (text, tags) = word.split('\t', maxsplit=1)
            doc[-1].append({'text': text, 'multi_ner': tags.split()})
    with open(filename, 'w', encoding='utf-8') as fout:
        json.dump(doc, fout)

def get_args(tmp_path, pretrain_file, train_json, dev_json, *extra_args):
    if False:
        return 10
    save_dir = tmp_path / 'models'
    args = ['--data_dir', str(tmp_path), '--wordvec_pretrain_file', pretrain_file, '--train_file', str(train_json), '--eval_file', str(dev_json), '--shorthand', 'en_test', '--max_steps', '100', '--eval_interval', '40', '--save_dir', str(save_dir)]
    args = args + list(extra_args)
    return args

def run_two_tag_training(pretrain_file, tmp_path, *extra_args, train_data=EN_TRAIN_2TAG):
    if False:
        return 10
    train_json = tmp_path / 'en_test.train.json'
    write_temp_2tag(train_json, train_data)
    dev_json = tmp_path / 'en_test.dev.json'
    write_temp_2tag(dev_json, EN_DEV_2TAG)
    args = get_args(tmp_path, pretrain_file, train_json, dev_json, *extra_args)
    return ner_tagger.main(args)

def test_basic_two_tag_training(pretrain_file, tmp_path):
    if False:
        for i in range(10):
            print('nop')
    trainer = run_two_tag_training(pretrain_file, tmp_path)
    assert len(trainer.model.tag_clfs) == 2
    assert len(trainer.model.crits) == 2
    assert len(trainer.vocab['tag'].lens()) == 2

def test_two_tag_training_backprop(pretrain_file, tmp_path):
    if False:
        i = 10
        return i + 15
    '\n    Test that the training is backproping both tags\n\n    We can do this by using the "finetune" mechanism and verifying\n    that the output tensors are different\n    '
    trainer = run_two_tag_training(pretrain_file, tmp_path)
    trainer.save(os.path.join(trainer.args['save_dir'], trainer.args['save_name']))
    new_trainer = run_two_tag_training(pretrain_file, tmp_path, '--finetune')
    assert len(trainer.model.tag_clfs) == 2
    assert len(new_trainer.model.tag_clfs) == 2
    for (old_clf, new_clf) in zip(trainer.model.tag_clfs, new_trainer.model.tag_clfs):
        assert not torch.allclose(old_clf.weight, new_clf.weight)

def test_two_tag_training_c2_backprop(pretrain_file, tmp_path):
    if False:
        return 10
    '\n    Test that the training is backproping only one tag if one column is blank\n\n    We can do this by using the "finetune" mechanism and verifying\n    that the output tensors are different in just the first column\n    '
    trainer = run_two_tag_training(pretrain_file, tmp_path)
    trainer.save(os.path.join(trainer.args['save_dir'], trainer.args['save_name']))
    new_trainer = run_two_tag_training(pretrain_file, tmp_path, '--finetune', train_data=EN_TRAIN_2TAG_EMPTY2)
    assert len(trainer.model.tag_clfs) == 2
    assert len(new_trainer.model.tag_clfs) == 2
    assert not torch.allclose(trainer.model.tag_clfs[0].weight, new_trainer.model.tag_clfs[0].weight)
    assert torch.allclose(trainer.model.tag_clfs[1].weight, new_trainer.model.tag_clfs[1].weight)

def test_connected_two_tag_training(pretrain_file, tmp_path):
    if False:
        for i in range(10):
            print('nop')
    trainer = run_two_tag_training(pretrain_file, tmp_path, '--connect_output_layers')
    assert len(trainer.model.tag_clfs) == 2
    assert len(trainer.model.crits) == 2
    assert len(trainer.vocab['tag'].lens()) == 2
    assert trainer.model.tag_clfs[1].weight.shape[1] == trainer.vocab['tag'].lens()[0] + trainer.model.tag_clfs[0].weight.shape[1]

def run_training(pretrain_file, tmp_path, *extra_args):
    if False:
        for i in range(10):
            print('nop')
    train_json = tmp_path / 'en_test.train.json'
    write_temp_file(train_json, EN_TRAIN_BIO)
    dev_json = tmp_path / 'en_test.dev.json'
    write_temp_file(dev_json, EN_DEV_BIO)
    args = get_args(tmp_path, pretrain_file, train_json, dev_json, *extra_args)
    return ner_tagger.main(args)

def test_train_model_gpu(pretrain_file, tmp_path):
    if False:
        return 10
    '\n    Briefly train an NER model (no expectation of correctness) and check that it is on the GPU\n    '
    trainer = run_training(pretrain_file, tmp_path)
    if not torch.cuda.is_available():
        warnings.warn('Cannot check that the NER model is on the GPU, since GPU is not available')
        return
    model = trainer.model
    device = next(model.parameters()).device
    assert str(device).startswith('cuda')

def test_train_model_cpu(pretrain_file, tmp_path):
    if False:
        print('Hello World!')
    '\n    Briefly train an NER model (no expectation of correctness) and check that it is on the GPU\n    '
    trainer = run_training(pretrain_file, tmp_path, '--cpu')
    model = trainer.model
    device = next(model.parameters()).device
    assert str(device).startswith('cpu')

def model_file_has_bert(filename):
    if False:
        print('Hello World!')
    checkpoint = torch.load(filename, lambda storage, loc: storage)
    return any((x.startswith('bert_model.') for x in checkpoint['model'].keys()))

def test_with_bert(pretrain_file, tmp_path):
    if False:
        print('Hello World!')
    trainer = run_training(pretrain_file, tmp_path, '--bert_model', 'hf-internal-testing/tiny-bert')
    model_file = os.path.join(trainer.args['save_dir'], trainer.args['save_name'])
    assert not model_file_has_bert(model_file)

def test_with_bert_finetune(pretrain_file, tmp_path):
    if False:
        i = 10
        return i + 15
    trainer = run_training(pretrain_file, tmp_path, '--bert_model', 'hf-internal-testing/tiny-bert', '--bert_finetune')
    model_file = os.path.join(trainer.args['save_dir'], trainer.args['save_name'])
    assert model_file_has_bert(model_file)
    foo_save_filename = os.path.join(tmp_path, 'foo_' + trainer.args['save_name'])
    bar_save_filename = os.path.join(tmp_path, 'bar_' + trainer.args['save_name'])
    trainer.save(foo_save_filename)
    assert model_file_has_bert(foo_save_filename)
    reloaded_trainer = Trainer(args=trainer.args, model_file=foo_save_filename)
    reloaded_trainer.save(bar_save_filename)
    assert model_file_has_bert(bar_save_filename)