"""
Test some simple conversions of NER bio files
"""
import pytest
import json
from stanza.models.common.doc import Document
from stanza.utils.datasets.ner.prepare_ner_file import process_dataset
BIO_1 = "\nJennifer\tB-PERSON\nSh'reyan\tI-PERSON\nhas\tO\nlovely\tO\nantennae\tO\n".strip()
BIO_2 = "\nbut\tO\nI\tO\ndon't\tO\nlike\tO\nthe\tO\nway\tO\nJennifer\tB-PERSON\ntreated\tO\nBeckett\tB-PERSON\non\tO\nthe\tO\nCerritos\tB-LOCATION\n".strip()

def check_json_file(doc, raw_text, expected_sentences, expected_tokens):
    if False:
        for i in range(10):
            print('nop')
    raw_sentences = raw_text.strip().split('\n\n')
    assert len(raw_sentences) == expected_sentences
    if isinstance(expected_tokens, int):
        expected_tokens = [expected_tokens]
    for (raw_sentence, expected_len) in zip(raw_sentences, expected_tokens):
        assert len(raw_sentence.strip().split('\n')) == expected_len
    assert len(doc.sentences) == expected_sentences
    for (sentence, expected_len) in zip(doc.sentences, expected_tokens):
        assert len(sentence.tokens) == expected_len
    for (sentence, raw_sentence) in zip(doc.sentences, raw_sentences):
        for (token, line) in zip(sentence.tokens, raw_sentence.strip().split('\n')):
            (word, tag) = line.strip().split()
            assert token.text == word
            assert token.ner == tag

def write_and_convert(tmp_path, raw_text):
    if False:
        while True:
            i = 10
    bio_file = tmp_path / 'test.bio'
    with open(bio_file, 'w', encoding='utf-8') as fout:
        fout.write(raw_text)
    json_file = tmp_path / 'json.bio'
    process_dataset(bio_file, json_file)
    with open(json_file) as fin:
        doc = Document(json.load(fin))
    return doc

def run_test(tmp_path, raw_text, expected_sentences, expected_tokens):
    if False:
        print('Hello World!')
    doc = write_and_convert(tmp_path, raw_text)
    check_json_file(doc, raw_text, expected_sentences, expected_tokens)

def test_simple(tmp_path):
    if False:
        while True:
            i = 10
    run_test(tmp_path, BIO_1, 1, 5)

def test_ner_at_end(tmp_path):
    if False:
        i = 10
        return i + 15
    run_test(tmp_path, BIO_2, 1, 12)

def test_two_sentences(tmp_path):
    if False:
        i = 10
        return i + 15
    raw_text = BIO_1 + '\n\n' + BIO_2
    run_test(tmp_path, raw_text, 2, [5, 12])