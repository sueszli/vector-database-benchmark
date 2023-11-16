import json
import os
import pytest
pytestmark = [pytest.mark.travis, pytest.mark.pipeline]
from stanza.models.common.doc import Document
from stanza.tests.ner.test_ner_training import write_temp_file, EN_TRAIN_BIO, EN_DEV_BIO
from stanza.utils.datasets.ner import combine_ner_datasets

def test_combine(tmp_path):
    if False:
        i = 10
        return i + 15
    '\n    Test that if we write two short datasets and combine them, we get back\n    one slightly longer dataset\n\n    To simplify matters, we just use the same input text with longer\n    amounts of text for each shard.\n    '
    SHARDS = ('train', 'dev', 'test')
    for (s_num, shard) in enumerate(SHARDS):
        t1_json = tmp_path / ('en_t1.%s.json' % shard)
        write_temp_file(t1_json, '\n\n'.join([EN_TRAIN_BIO] * (s_num + 1)))
        t2_json = tmp_path / ('en_t2.%s.json' % shard)
        write_temp_file(t2_json, '\n\n'.join([EN_DEV_BIO] * (s_num + 1)))
    args = ['--output_dataset', 'en_c', 'en_t1', 'en_t2', '--input_dir', str(tmp_path), '--output_dir', str(tmp_path)]
    combine_ner_datasets.main(args)
    for (s_num, shard) in enumerate(SHARDS):
        filename = tmp_path / ('en_c.%s.json' % shard)
        assert os.path.exists(filename)
        with open(filename, encoding='utf-8') as fin:
            doc = Document(json.load(fin))
            assert len(doc.sentences) == (s_num + 1) * 3