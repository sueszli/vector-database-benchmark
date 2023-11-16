"""
Runs a few tests on the split_wikiner file
"""
import os
import tempfile
import pytest
from stanza.utils.datasets.ner import split_wikiner
from stanza.tests import *
pytestmark = [pytest.mark.pipeline, pytest.mark.travis]
FBK_SAMPLE = '\nIl\tO\nPapa\tO\nsi\tO\naggrava\tO\n\nLe\tO\ncondizioni\tO\ndi\tO\n\nPapa\tO\nGiovanni\tPER\nPaolo\tPER\nII\tPER\nsi\tO\n\nsono\tO\naggravate\tO\nin\tO\nil\tO\ncorso\tO\n\ndi\tO\nla\tO\ngiornata\tO\ndi\tO\ngiovedì\tO\n.\tO\n\nIl\tO\nportavoce\tO\nNavarro\tPER\nValls\tPER\n\nha\tO\ndichiarato\tO\nche\tO\n\nil\tO\nSanto\tO\nPadre\tO\n\nin\tO\nla\tO\ngiornata\tO\n\ndi\tO\noggi\tO\nè\tO\nstato\tO\n\ncolpito\tO\nda\tO\nuna\tO\naffezione\tO\n\naltamente\tO\nfebbrile\tO\nprovocata\tO\nda\tO\nuna\tO\n\ninfezione\tO\ndocumentata\tO\n\ndi\tO\nle\tO\nvie\tO\nurinarie\tO\n.\tO\n\nA\tO\nil\tO\nmomento\tO\n\nnon\tO\nè\tO\nprevisto\tO\nil\tO\nricovero\tO\n\na\tO\nil\tO\nPoliclinico\tLOC\nGemelli\tLOC\n,\tO\n\ncome\tO\nha\tO\nprecisato\tO\nil\tO\n\nresponsabile\tO\ndi\tO\nil\tO\ndipartimento\tO\n\ndi\tO\nemergenza\tO\nprofessor\tO\nRodolfo\tPER\nProietti\tPER\n.\tO\n'

def test_read_sentences():
    if False:
        while True:
            i = 10
    with tempfile.TemporaryDirectory() as tempdir:
        raw_filename = os.path.join(tempdir, 'raw.tsv')
        with open(raw_filename, 'w') as fout:
            fout.write(FBK_SAMPLE)
        sentences = split_wikiner.read_sentences(raw_filename, 'utf-8')
        assert len(sentences) == 20
        text = [['\t'.join(word) for word in sent] for sent in sentences]
        text = ['\n'.join(sent) for sent in text]
        text = '\n\n'.join(text)
        assert FBK_SAMPLE.strip() == text

def test_write_sentences():
    if False:
        while True:
            i = 10
    with tempfile.TemporaryDirectory() as tempdir:
        raw_filename = os.path.join(tempdir, 'raw.tsv')
        with open(raw_filename, 'w') as fout:
            fout.write(FBK_SAMPLE)
        sentences = split_wikiner.read_sentences(raw_filename, 'utf-8')
        copy_filename = os.path.join(tempdir, 'copy.tsv')
        split_wikiner.write_sentences_to_file(sentences, copy_filename)
        sent2 = split_wikiner.read_sentences(raw_filename, 'utf-8')
        assert sent2 == sentences

def run_split_wikiner(expected_train=14, expected_dev=3, expected_test=3, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Runs a test using various parameters to check the results of the splitting process\n    '
    with tempfile.TemporaryDirectory() as indir:
        raw_filename = os.path.join(indir, 'raw.tsv')
        with open(raw_filename, 'w') as fout:
            fout.write(FBK_SAMPLE)
        with tempfile.TemporaryDirectory() as outdir:
            split_wikiner.split_wikiner(outdir, raw_filename, **kwargs)
            train_file = os.path.join(outdir, 'it_fbk.train.bio')
            dev_file = os.path.join(outdir, 'it_fbk.dev.bio')
            test_file = os.path.join(outdir, 'it_fbk.test.bio')
            assert os.path.exists(train_file)
            assert os.path.exists(dev_file)
            if kwargs['test_section']:
                assert os.path.exists(test_file)
            else:
                assert not os.path.exists(test_file)
            train_sent = split_wikiner.read_sentences(train_file, 'utf-8')
            dev_sent = split_wikiner.read_sentences(dev_file, 'utf-8')
            assert len(train_sent) == expected_train
            assert len(dev_sent) == expected_dev
            if kwargs['test_section']:
                test_sent = split_wikiner.read_sentences(test_file, 'utf-8')
                assert len(test_sent) == expected_test
            else:
                test_sent = []
            if kwargs['shuffle']:
                orig_sents = sorted(split_wikiner.read_sentences(raw_filename, 'utf-8'))
                split_sents = sorted(train_sent + dev_sent + test_sent)
            else:
                orig_sents = split_wikiner.read_sentences(raw_filename, 'utf-8')
                split_sents = train_sent + dev_sent + test_sent
            assert orig_sents == split_sents

def test_no_shuffle_split():
    if False:
        i = 10
        return i + 15
    run_split_wikiner(prefix='it_fbk', shuffle=False, test_section=True)

def test_shuffle_split():
    if False:
        print('Hello World!')
    run_split_wikiner(prefix='it_fbk', shuffle=True, test_section=True)

def test_resize():
    if False:
        for i in range(10):
            print('nop')
    run_split_wikiner(expected_train=12, expected_dev=2, expected_test=6, train_fraction=0.6, dev_fraction=0.1, prefix='it_fbk', shuffle=True, test_section=True)

def test_no_test_split():
    if False:
        while True:
            i = 10
    run_split_wikiner(expected_train=17, train_fraction=0.85, prefix='it_fbk', shuffle=False, test_section=False)