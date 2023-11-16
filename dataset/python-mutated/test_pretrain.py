import os
import tempfile
import pytest
import numpy as np
import torch
from stanza.models.common import pretrain
from stanza.models.common.vocab import UNK_ID
from stanza.tests import *
pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

def check_vocab(vocab):
    if False:
        print('Hello World!')
    assert len(vocab) == 7
    assert 'unban' in vocab
    assert 'mox' in vocab
    assert 'opal' in vocab

def check_embedding(emb, unk=False):
    if False:
        return 10
    expected = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
    if unk:
        expected[UNK_ID] = -1
    np.testing.assert_allclose(emb, expected)

def check_pretrain(pt):
    if False:
        i = 10
        return i + 15
    check_vocab(pt.vocab)
    check_embedding(pt.emb)

def test_text_pretrain():
    if False:
        for i in range(10):
            print('nop')
    pt = pretrain.Pretrain(vec_filename=f'{TEST_WORKING_DIR}/in/tiny_emb.txt', save_to_file=False)
    check_pretrain(pt)

def test_xz_pretrain():
    if False:
        return 10
    pt = pretrain.Pretrain(vec_filename=f'{TEST_WORKING_DIR}/in/tiny_emb.xz', save_to_file=False)
    check_pretrain(pt)

def test_gz_pretrain():
    if False:
        i = 10
        return i + 15
    pt = pretrain.Pretrain(vec_filename=f'{TEST_WORKING_DIR}/in/tiny_emb.gz', save_to_file=False)
    check_pretrain(pt)

def test_zip_pretrain():
    if False:
        while True:
            i = 10
    pt = pretrain.Pretrain(vec_filename=f'{TEST_WORKING_DIR}/in/tiny_emb.zip', save_to_file=False)
    check_pretrain(pt)

def test_csv_pretrain():
    if False:
        for i in range(10):
            print('nop')
    pt = pretrain.Pretrain(csv_filename=f'{TEST_WORKING_DIR}/in/tiny_emb.csv', save_to_file=False)
    check_pretrain(pt)

def test_resave_pretrain():
    if False:
        while True:
            i = 10
    '\n    Test saving a pretrain and then loading from the existing file\n    '
    test_pt_file = tempfile.NamedTemporaryFile(dir=f'{TEST_WORKING_DIR}/out', suffix='.pt', delete=False)
    try:
        test_pt_file.close()
        pt = pretrain.Pretrain(filename=test_pt_file.name, vec_filename=f'{TEST_WORKING_DIR}/in/tiny_emb.xz')
        check_pretrain(pt)
        pt2 = pretrain.Pretrain(filename=test_pt_file.name, vec_filename=f'unban_mox_opal')
        check_pretrain(pt2)
        pt3 = torch.load(test_pt_file.name)
        check_embedding(pt3['emb'])
    finally:
        os.unlink(test_pt_file.name)
SPACE_PRETRAIN = '\n3 4\nunban mox 1 2 3 4\nopal 5 6 7 8\nfoo 9 10 11 12\n'.strip()

def test_whitespace():
    if False:
        while True:
            i = 10
    '\n    Test reading a pretrain with an ascii space in it\n\n    The vocab word with a space in it should have the correct number\n    of dimensions read, with the space converted to nbsp\n    '
    test_txt_file = tempfile.NamedTemporaryFile(dir=f'{TEST_WORKING_DIR}/out', suffix='.txt', delete=False)
    try:
        test_txt_file.write(SPACE_PRETRAIN.encode())
        test_txt_file.close()
        pt = pretrain.Pretrain(vec_filename=test_txt_file.name, save_to_file=False)
        check_embedding(pt.emb)
        assert 'unban\xa0mox' in pt.vocab
        assert 'unban mox' in pt.vocab
    finally:
        os.unlink(test_txt_file.name)
NO_HEADER_PRETRAIN = '\nunban 1 2 3 4\nmox 5 6 7 8\nopal 9 10 11 12\n'.strip()

def test_no_header():
    if False:
        while True:
            i = 10
    '\n    Check loading a pretrain with no rows,cols header\n    '
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as tmpdir:
        filename = os.path.join(tmpdir, 'tiny.txt')
        with open(filename, 'w', encoding='utf-8') as fout:
            fout.write(NO_HEADER_PRETRAIN)
        pt = pretrain.Pretrain(vec_filename=filename, save_to_file=False)
        check_embedding(pt.emb)
UNK_PRETRAIN = '\nunban 1 2 3 4\nmox 5 6 7 8\nopal 9 10 11 12\n<unk> -1 -1 -1 -1\n'.strip()

def test_no_header():
    if False:
        print('Hello World!')
    '\n    Check loading a pretrain with <unk> at the end, like GloVe does\n    '
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as tmpdir:
        filename = os.path.join(tmpdir, 'tiny.txt')
        with open(filename, 'w', encoding='utf-8') as fout:
            fout.write(UNK_PRETRAIN)
        pt = pretrain.Pretrain(vec_filename=filename, save_to_file=False)
        check_embedding(pt.emb, unk=True)