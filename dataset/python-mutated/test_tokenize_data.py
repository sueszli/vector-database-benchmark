"""
Very simple test of the mwt counting functionality in tokenization/data.py

TODO: could add a bunch more simple tests, including tests of reading
the data from a temp file, for example
"""
import pytest
import tempfile
import numpy as np
import stanza
from stanza import Pipeline
from stanza.tests import *
from stanza.models.tokenization.data import DataLoader, NUMERIC_RE
pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

def write_tokenizer_input(test_dir, raw_text, labels):
    if False:
        while True:
            i = 10
    '\n    Writes raw_text and labels to randomly named files in test_dir\n\n    Note that the tempfiles are not set to automatically clean up.\n    This will not be a problem if you put them in a tempdir.\n    '
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=test_dir, delete=False) as fout:
        txt_file = fout.name
        fout.write(raw_text)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=test_dir, delete=False) as fout:
        label_file = fout.name
        fout.write(labels)
    return (txt_file, label_file)
NO_MWT_TEXT = 'Sehr gute Beratung, schnelle Behebung der Probleme'
NO_MWT_LABELS = '00010000100000000110000000010000000010001000000002'
MWT_TEXT = ' Die Kosten sind definitiv auch im Rahmen.'
MWT_LABELS = '000100000010000100000000010000100300000012'
FAKE_PROPERTIES = {'lang': 'de', 'feat_funcs': ('space_before', 'capitalized'), 'max_seqlen': 300, 'use_dictionary': False}

def test_has_mwt():
    if False:
        for i in range(10):
            print('nop')
    '\n    One dataset has no mwt, the other does\n    '
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as test_dir:
        (txt_file, label_file) = write_tokenizer_input(test_dir, NO_MWT_TEXT, NO_MWT_LABELS)
        data = DataLoader(args=FAKE_PROPERTIES, input_files={'txt': txt_file, 'label': label_file})
        assert not data.has_mwt()
        (txt_file, label_file) = write_tokenizer_input(test_dir, MWT_TEXT, MWT_LABELS)
        data = DataLoader(args=FAKE_PROPERTIES, input_files={'txt': txt_file, 'label': label_file})
        assert data.has_mwt()

@pytest.fixture(scope='module')
def tokenizer():
    if False:
        for i in range(10):
            print('nop')
    pipeline = Pipeline('en', dir=TEST_MODELS_DIR, download_method=None, processors='tokenize')
    tokenizer = pipeline.processors['tokenize']
    return tokenizer

@pytest.fixture(scope='module')
def zhtok():
    if False:
        print('Hello World!')
    pipeline = Pipeline('zh-hans', dir=TEST_MODELS_DIR, download_method=None, processors='tokenize')
    tokenizer = pipeline.processors['tokenize']
    return tokenizer
EXPECTED_TWO_NL_RAW = [[('T', 0), ('h', 0), ('i', 0), ('s', 0), (' ', 0), ('i', 0), ('s', 0), (' ', 0), ('a', 0), (' ', 0), ('t', 0), ('e', 0), ('s', 0), ('t', 0)], [('f', 0), ('o', 0), ('o', 0)]]
EXPECTED_ONE_NL_RAW = [[('T', 0), ('h', 0), ('i', 0), ('s', 0), (' ', 0), ('i', 0), ('s', 0), (' ', 0), ('a', 0), (' ', 0), ('t', 0), ('e', 0), ('s', 0), ('t', 0), (' ', 0), ('f', 0), ('o', 0), ('o', 0)]]
EXPECTED_SKIP_NL_RAW = [[('T', 0), ('h', 0), ('i', 0), ('s', 0), (' ', 0), ('i', 0), ('s', 0), (' ', 0), ('a', 0), (' ', 0), ('t', 0), ('e', 0), ('s', 0), ('t', 0), ('f', 0), ('o', 0), ('o', 0)]]

def test_convert_units_raw_text(tokenizer):
    if False:
        while True:
            i = 10
    '\n    Tests converting a couple small segments to units\n    '
    raw_text = 'This is a      test\n\nfoo'
    batches = DataLoader(tokenizer.config, input_text=raw_text, vocab=tokenizer.vocab, evaluation=True, dictionary=tokenizer.trainer.dictionary)
    assert batches.data == EXPECTED_TWO_NL_RAW
    raw_text = 'This is a      test\nfoo'
    batches = DataLoader(tokenizer.config, input_text=raw_text, vocab=tokenizer.vocab, evaluation=True, dictionary=tokenizer.trainer.dictionary)
    assert batches.data == EXPECTED_ONE_NL_RAW
    skip_newline_config = dict(tokenizer.config)
    skip_newline_config['skip_newline'] = True
    batches = DataLoader(skip_newline_config, input_text=raw_text, vocab=tokenizer.vocab, evaluation=True, dictionary=tokenizer.trainer.dictionary)
    assert batches.data == EXPECTED_SKIP_NL_RAW
EXPECTED_TWO_NL_FILE = [[('T', 0), ('h', 0), ('i', 0), ('s', 0), (' ', 0), ('i', 0), ('s', 0), (' ', 0), ('a', 0), (' ', 0), ('t', 0), ('e', 0), ('s', 0), ('t', 0), ('.', 1)], [('f', 0), ('o', 0), ('o', 0)]]
EXPECTED_TWO_NL_FILE_LABELS = [np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.int32), np.array([0, 0, 0], dtype=np.int32)]
EXPECTED_ONE_NL_FILE = [[('T', 0), ('h', 0), ('i', 0), ('s', 0), (' ', 0), ('i', 0), ('s', 0), (' ', 0), ('a', 0), (' ', 0), ('t', 0), ('e', 0), ('s', 0), ('t', 0), ('.', 1), (' ', 0), ('f', 0), ('o', 0), ('o', 0)]]
EXPECTED_ONE_NL_FILE_LABELS = [np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.int32)]
EXPECTED_SKIP_NL_FILE = [[('T', 0), ('h', 0), ('i', 0), ('s', 0), (' ', 0), ('i', 0), ('s', 0), (' ', 0), ('a', 0), (' ', 0), ('t', 0), ('e', 0), ('s', 0), ('t', 0), ('.', 1), ('f', 0), ('o', 0), ('o', 0)]]
EXPECTED_SKIP_NL_FILE_LABELS = [np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=np.int32)]

def check_labels(labels, expected_labels):
    if False:
        return 10
    assert len(labels) == len(expected_labels)
    for (label, expected) in zip(labels, expected_labels):
        assert np.array_equiv(label, expected)

def test_convert_units_file(tokenizer):
    if False:
        i = 10
        return i + 15
    '\n    Tests reading some text from a file and converting that to units\n    '
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as test_dir:
        labels = '00000000000000000001\n\n000\n\n'
        raw_text = 'This is a      test.\n\nfoo\n\n'
        (txt_file, label_file) = write_tokenizer_input(test_dir, raw_text, labels)
        batches = DataLoader(tokenizer.config, input_files={'txt': txt_file, 'label': label_file}, vocab=tokenizer.vocab, evaluation=True, dictionary=tokenizer.trainer.dictionary)
        assert batches.data == EXPECTED_TWO_NL_FILE
        check_labels(batches.labels(), EXPECTED_TWO_NL_FILE_LABELS)
        labels = '000000000000000000010000\n\n'
        raw_text = 'This is a      test.\nfoo\n\n'
        (txt_file, label_file) = write_tokenizer_input(test_dir, raw_text, labels)
        batches = DataLoader(tokenizer.config, input_files={'txt': txt_file, 'label': label_file}, vocab=tokenizer.vocab, evaluation=True, dictionary=tokenizer.trainer.dictionary)
        assert batches.data == EXPECTED_ONE_NL_FILE
        check_labels(batches.labels(), EXPECTED_ONE_NL_FILE_LABELS)
        skip_newline_config = dict(tokenizer.config)
        skip_newline_config['skip_newline'] = True
        labels = '000000000000000000010000\n\n'
        raw_text = 'This is a      test.\nfoo\n\n'
        (txt_file, label_file) = write_tokenizer_input(test_dir, raw_text, labels)
        batches = DataLoader(skip_newline_config, input_files={'txt': txt_file, 'label': label_file}, vocab=tokenizer.vocab, evaluation=True, dictionary=tokenizer.trainer.dictionary)
        assert batches.data == EXPECTED_SKIP_NL_FILE
        check_labels(batches.labels(), EXPECTED_SKIP_NL_FILE_LABELS)

def test_dictionary(zhtok):
    if False:
        while True:
            i = 10
    '\n    Tests some features of the zh tokenizer dictionary\n\n    The expectation is that the Chinese tokenizer will be serialized with a dictionary\n    (if it ever gets serialized without, this test will warn us!)\n    '
    assert zhtok.trainer.lexicon is not None
    assert zhtok.trainer.dictionary is not None
    assert '老师' in zhtok.trainer.lexicon
    assert '蛋白质' in zhtok.trainer.lexicon
    assert '蛋白' in zhtok.trainer.dictionary['prefixes']
    assert '蛋' in zhtok.trainer.dictionary['prefixes']
    assert '白质' in zhtok.trainer.dictionary['suffixes']
    assert '质' in zhtok.trainer.dictionary['suffixes']

def test_dictionary_feats(zhtok):
    if False:
        print('Hello World!')
    '\n    Test the results of running a sentence into the dictionary featurizer\n    '
    raw_text = '我想吃蛋白质'
    batches = DataLoader(zhtok.config, input_text=raw_text, vocab=zhtok.vocab, evaluation=True, dictionary=zhtok.trainer.dictionary)
    data = batches.data
    assert len(data) == 1
    assert len(data[0]) == 6
    expected_features = [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0]]
    for (i, expected) in enumerate(expected_features):
        dict_features = batches.extract_dict_feat(data[0], i)
        assert dict_features == expected

def test_numeric_re():
    if False:
        while True:
            i = 10
    '\n    Test the "is numeric" function\n\n    This function is entirely based on an RE in data.py\n    '
    matches = ['57', '135245345', '12535.', '852358.458345', '435345...345345', '111,,,111,,,111,,,111', '5318008', '５', '๕']
    not_matches = ['.4', '54353a', '5453 35345', 'aaa143234', 'a,a,a,a', "sh'reyan", 'asdaf786876asdfasdf', '', '11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111a']
    for x in matches:
        assert NUMERIC_RE.match(x) is not None
    for x in not_matches:
        assert NUMERIC_RE.match(x) is None