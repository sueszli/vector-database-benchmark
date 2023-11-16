import pytest
from spacy.tokens import Span, SpanGroup
from spacy.tokens._dict_proxies import SpanGroups

@pytest.mark.issue(10685)
def test_issue10685(en_tokenizer):
    if False:
        print('Hello World!')
    'Test `SpanGroups` de/serialization'
    doc = en_tokenizer('Will it blend?')
    assert len(doc.spans) == 0
    doc.spans.from_bytes(doc.spans.to_bytes())
    assert len(doc.spans) == 0
    doc.spans['test'] = SpanGroup(doc, name='test', spans=[doc[0:1]])
    doc.spans['test2'] = SpanGroup(doc, name='test', spans=[doc[1:2]])

    def assert_spangroups():
        if False:
            return 10
        assert len(doc.spans) == 2
        assert doc.spans['test'].name == 'test'
        assert doc.spans['test2'].name == 'test'
        assert list(doc.spans['test']) == [doc[0:1]]
        assert list(doc.spans['test2']) == [doc[1:2]]
    assert_spangroups()
    doc.spans.from_bytes(doc.spans.to_bytes())
    assert_spangroups()

def test_span_groups_serialization_mismatches(en_tokenizer):
    if False:
        while True:
            i = 10
    'Test the serialization of multiple mismatching `SpanGroups` keys and `SpanGroup.name`s'
    doc = en_tokenizer('How now, brown cow?')
    groups = doc.spans
    groups['key1'] = SpanGroup(doc, name='key1', spans=[doc[0:1], doc[1:2]])
    groups['key2'] = SpanGroup(doc, name='too', spans=[doc[3:4], doc[4:5]])
    groups['key3'] = SpanGroup(doc, name='too', spans=[doc[1:2], doc[0:1]])
    groups['key4'] = SpanGroup(doc, name='key4', spans=[doc[0:1]])
    groups['key5'] = SpanGroup(doc, name='key4', spans=[doc[0:1]])
    sg6 = SpanGroup(doc, name='key6', spans=[doc[0:1]])
    groups['key6'] = sg6
    groups['key7'] = sg6
    sg8 = SpanGroup(doc, name='also', spans=[doc[1:2]])
    groups['key8'] = sg8
    groups['key9'] = sg8
    regroups = SpanGroups(doc).from_bytes(groups.to_bytes())
    assert regroups.keys() == groups.keys()
    for (key, regroup) in regroups.items():
        assert regroup.name == groups[key].name
        assert list(regroup) == list(groups[key])

@pytest.mark.parametrize('spans_bytes,doc_text,expected_spangroups,expected_warning', [(b'\x90', '', {}, False), (b'\x91\xc4C\x83\xa4name\xa4test\xa5attrs\x80\xa5spans\x91\xc4(\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x04', 'Will it blend?', {'test': {'name': 'test', 'spans': [(0, 1)]}}, False), (b'\x92\xc4C\x83\xa4name\xa4test\xa5attrs\x80\xa5spans\x91\xc4(\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x04\xc4C\x83\xa4name\xa4test\xa5attrs\x80\xa5spans\x91\xc4(\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x05\x00\x00\x00\x07', 'Will it blend?', {'test': {'name': 'test', 'spans': [(1, 2)]}}, True), (b'\x95\xc4m\x83\xa4name\xa4key1\xa5attrs\x80\xa5spans\x92\xc4(\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x03\xc4(\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x04\x00\x00\x00\x07\xc4l\x83\xa4name\xa3too\xa5attrs\x80\xa5spans\x92\xc4(\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\t\x00\x00\x00\x0e\xc4(\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x05\x00\x00\x00\x0f\x00\x00\x00\x12\xc4l\x83\xa4name\xa3too\xa5attrs\x80\xa5spans\x92\xc4(\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x04\x00\x00\x00\x07\xc4(\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x03\xc4C\x83\xa4name\xa4key4\xa5attrs\x80\xa5spans\x91\xc4(\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x03\xc4C\x83\xa4name\xa4key4\xa5attrs\x80\xa5spans\x91\xc4(\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x03', 'How now, brown cow?', {'key1': {'name': 'key1', 'spans': [(0, 1), (1, 2)]}, 'too': {'name': 'too', 'spans': [(1, 2), (0, 1)]}, 'key4': {'name': 'key4', 'spans': [(0, 1)]}}, True)])
def test_deserialize_span_groups_compat(en_tokenizer, spans_bytes, doc_text, expected_spangroups, expected_warning):
    if False:
        while True:
            i = 10
    'Test backwards-compatibility of `SpanGroups` deserialization.\n    This uses serializations (bytes) from a prior version of spaCy (before 3.3.1).\n\n    spans_bytes (bytes): Serialized `SpanGroups` object.\n    doc_text (str): Doc text.\n    expected_spangroups (dict):\n        Dict mapping every expected (after deserialization) `SpanGroups` key\n        to a SpanGroup\'s "args", where a SpanGroup\'s args are given as a dict:\n          {"name": span_group.name,\n           "spans": [(span0.start, span0.end), ...]}\n    expected_warning (bool): Whether a warning is to be expected from .from_bytes()\n        --i.e. if more than 1 SpanGroup has the same .name within the `SpanGroups`.\n    '
    doc = en_tokenizer(doc_text)
    if expected_warning:
        with pytest.warns(UserWarning):
            doc.spans.from_bytes(spans_bytes)
    else:
        doc.spans.from_bytes(spans_bytes)
    assert doc.spans.keys() == expected_spangroups.keys()
    for (name, spangroup_args) in expected_spangroups.items():
        assert doc.spans[name].name == spangroup_args['name']
        spans = [Span(doc, start, end) for (start, end) in spangroup_args['spans']]
        assert list(doc.spans[name]) == spans

def test_span_groups_serialization(en_tokenizer):
    if False:
        for i in range(10):
            print('nop')
    doc = en_tokenizer('0 1 2 3 4 5 6')
    span_groups = SpanGroups(doc)
    spans = [doc[0:2], doc[1:3]]
    sg1 = SpanGroup(doc, spans=spans)
    span_groups['key1'] = sg1
    span_groups['key2'] = sg1
    span_groups['key3'] = []
    reloaded_span_groups = SpanGroups(doc).from_bytes(span_groups.to_bytes())
    assert span_groups.keys() == reloaded_span_groups.keys()
    for (key, value) in span_groups.items():
        assert all((span == reloaded_span for (span, reloaded_span) in zip(span_groups[key], reloaded_span_groups[key])))