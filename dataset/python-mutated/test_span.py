import numpy
import pytest
from numpy.testing import assert_array_equal
from thinc.api import get_current_ops
from spacy.attrs import LENGTH, ORTH
from spacy.lang.en import English
from spacy.tokens import Doc, Span, Token
from spacy.util import filter_spans
from spacy.vocab import Vocab
from ..util import add_vecs_to_vocab
from .test_underscore import clean_underscore

@pytest.fixture
def doc(en_tokenizer):
    if False:
        i = 10
        return i + 15
    text = 'This is a sentence. This is another sentence. And a third.'
    heads = [1, 1, 3, 1, 1, 6, 6, 8, 6, 6, 12, 12, 12, 12]
    deps = ['nsubj', 'ROOT', 'det', 'attr', 'punct', 'nsubj', 'ROOT', 'det', 'attr', 'punct', 'ROOT', 'det', 'npadvmod', 'punct']
    ents = ['O', 'O', 'B-ENT', 'I-ENT', 'I-ENT', 'I-ENT', 'I-ENT', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
    tokens = en_tokenizer(text)
    lemmas = [t.text for t in tokens]
    spaces = [bool(t.whitespace_) for t in tokens]
    return Doc(tokens.vocab, words=[t.text for t in tokens], spaces=spaces, heads=heads, deps=deps, ents=ents, lemmas=lemmas)

@pytest.fixture
def doc_not_parsed(en_tokenizer):
    if False:
        print('Hello World!')
    text = 'This is a sentence. This is another sentence. And a third.'
    tokens = en_tokenizer(text)
    doc = Doc(tokens.vocab, words=[t.text for t in tokens])
    return doc

@pytest.mark.issue(1537)
def test_issue1537():
    if False:
        i = 10
        return i + 15
    "Test that Span.as_doc() doesn't segfault."
    string = 'The sky is blue . The man is pink . The dog is purple .'
    doc = Doc(Vocab(), words=string.split())
    doc[0].sent_start = True
    for word in doc[1:]:
        if word.nbor(-1).text == '.':
            word.sent_start = True
        else:
            word.sent_start = False
    sents = list(doc.sents)
    sent0 = sents[0].as_doc()
    sent1 = sents[1].as_doc()
    assert isinstance(sent0, Doc)
    assert isinstance(sent1, Doc)

@pytest.mark.issue(1612)
def test_issue1612(en_tokenizer):
    if False:
        print('Hello World!')
    'Test that span.orth_ is identical to span.text'
    doc = en_tokenizer('The black cat purrs.')
    span = doc[1:3]
    assert span.orth_ == span.text

@pytest.mark.issue(3199)
def test_issue3199():
    if False:
        return 10
    "Test that Span.noun_chunks works correctly if no noun chunks iterator\n    is available. To make this test future-proof, we're constructing a Doc\n    with a new Vocab here and a parse tree to make sure the noun chunks run.\n    "
    words = ['This', 'is', 'a', 'sentence']
    doc = Doc(Vocab(), words=words, heads=[0] * len(words), deps=['dep'] * len(words))
    with pytest.raises(NotImplementedError):
        list(doc[0:3].noun_chunks)

@pytest.mark.issue(5152)
def test_issue5152():
    if False:
        return 10
    nlp = English()
    text = nlp('Talk about being boring!')
    text_var = nlp('Talk of being boring!')
    y = nlp('Let')
    span = text[0:3]
    span_2 = text[0:3]
    span_3 = text_var[0:3]
    token = y[0]
    with pytest.warns(UserWarning):
        assert span.similarity(token) == 0.0
    assert span.similarity(span_2) == 1.0
    with pytest.warns(UserWarning):
        assert span_2.similarity(span_3) < 1.0

@pytest.mark.issue(6755)
def test_issue6755(en_tokenizer):
    if False:
        print('Hello World!')
    doc = en_tokenizer('This is a magnificent sentence.')
    span = doc[:0]
    assert span.text_with_ws == ''
    assert span.text == ''

@pytest.mark.parametrize('sentence, start_idx,end_idx,label', [('Welcome to Mumbai, my friend', 11, 17, 'GPE')])
@pytest.mark.issue(6815)
def test_issue6815_1(sentence, start_idx, end_idx, label):
    if False:
        for i in range(10):
            print('nop')
    nlp = English()
    doc = nlp(sentence)
    span = doc[:].char_span(start_idx, end_idx, label=label)
    assert span.label_ == label

@pytest.mark.parametrize('sentence, start_idx,end_idx,kb_id', [('Welcome to Mumbai, my friend', 11, 17, 5)])
@pytest.mark.issue(6815)
def test_issue6815_2(sentence, start_idx, end_idx, kb_id):
    if False:
        for i in range(10):
            print('nop')
    nlp = English()
    doc = nlp(sentence)
    span = doc[:].char_span(start_idx, end_idx, kb_id=kb_id)
    assert span.kb_id == kb_id

@pytest.mark.parametrize('sentence, start_idx,end_idx,vector', [('Welcome to Mumbai, my friend', 11, 17, numpy.array([0.1, 0.2, 0.3]))])
@pytest.mark.issue(6815)
def test_issue6815_3(sentence, start_idx, end_idx, vector):
    if False:
        while True:
            i = 10
    nlp = English()
    doc = nlp(sentence)
    span = doc[:].char_span(start_idx, end_idx, vector=vector)
    assert (span.vector == vector).all()

@pytest.mark.parametrize('i_sent,i,j,text', [(0, 0, len('This is a'), 'This is a'), (1, 0, len('This is another'), 'This is another'), (2, len('And '), len('And ') + len('a third'), 'a third'), (0, 1, 2, None)])
def test_char_span(doc, i_sent, i, j, text):
    if False:
        while True:
            i = 10
    sents = list(doc.sents)
    span = sents[i_sent].char_span(i, j)
    if not text:
        assert not span
    else:
        assert span.text == text

def test_char_span_attributes(doc):
    if False:
        return 10
    label = 'LABEL'
    kb_id = 'KB_ID'
    span_id = 'SPAN_ID'
    span1 = doc.char_span(20, 45, label=label, kb_id=kb_id, span_id=span_id)
    span2 = doc[1:].char_span(15, 40, label=label, kb_id=kb_id, span_id=span_id)
    assert span1.text == span2.text
    assert span1.label_ == span2.label_ == label
    assert span1.kb_id_ == span2.kb_id_ == kb_id
    assert span1.id_ == span2.id_ == span_id

def test_spans_sent_spans(doc):
    if False:
        return 10
    sents = list(doc.sents)
    assert sents[0].start == 0
    assert sents[0].end == 5
    assert len(sents) == 3
    assert sum((len(sent) for sent in sents)) == len(doc)

def test_spans_root(doc):
    if False:
        return 10
    span = doc[2:4]
    assert len(span) == 2
    assert span.text == 'a sentence'
    assert span.root.text == 'sentence'
    assert span.root.head.text == 'is'

def test_spans_string_fn(doc):
    if False:
        while True:
            i = 10
    span = doc[0:4]
    assert len(span) == 4
    assert span.text == 'This is a sentence'

def test_spans_root2(en_tokenizer):
    if False:
        for i in range(10):
            print('nop')
    text = 'through North and South Carolina'
    heads = [0, 4, 1, 1, 0]
    deps = ['dep'] * len(heads)
    tokens = en_tokenizer(text)
    doc = Doc(tokens.vocab, words=[t.text for t in tokens], heads=heads, deps=deps)
    assert doc[-2:].root.text == 'Carolina'

def test_spans_span_sent(doc, doc_not_parsed):
    if False:
        while True:
            i = 10
    'Test span.sent property'
    assert len(list(doc.sents))
    assert doc[:2].sent.root.text == 'is'
    assert doc[:2].sent.text == 'This is a sentence.'
    assert doc[6:7].sent.root.left_edge.text == 'This'
    assert doc[0:len(doc)].sent == list(doc.sents)[0]
    assert list(doc[0:len(doc)].sents) == list(doc.sents)
    with pytest.raises(ValueError):
        doc_not_parsed[:2].sent
    doc_not_parsed[0].is_sent_start = True
    doc_not_parsed[5].is_sent_start = True
    assert doc_not_parsed[1:3].sent == doc_not_parsed[0:5]
    assert doc_not_parsed[10:14].sent == doc_not_parsed[5:]

@pytest.mark.parametrize('start,end,expected_sentence', [(0, 14, 'This is'), (1, 4, 'This is'), (0, 2, 'This is'), (0, 1, 'This is'), (10, 14, 'And a'), (12, 14, 'third.'), (1, 1, 'This is')])
def test_spans_span_sent_user_hooks(doc, start, end, expected_sentence):
    if False:
        i = 10
        return i + 15

    def user_hook(doc):
        if False:
            return 10
        return [doc[ii:ii + 2] for ii in range(0, len(doc), 2)]
    doc.user_hooks['sents'] = user_hook
    assert doc[start:end].sent.text == expected_sentence
    doc.user_span_hooks['sent'] = lambda x: x
    assert doc[start:end].sent == doc[start:end]

def test_spans_lca_matrix(en_tokenizer):
    if False:
        return 10
    "Test span's lca matrix generation"
    tokens = en_tokenizer('the lazy dog slept')
    doc = Doc(tokens.vocab, words=[t.text for t in tokens], heads=[2, 2, 3, 3], deps=['dep'] * 4)
    lca = doc[:2].get_lca_matrix()
    assert lca.shape == (2, 2)
    assert lca[0, 0] == 0
    assert lca[0, 1] == -1
    assert lca[1, 0] == -1
    assert lca[1, 1] == 1
    lca = doc[1:].get_lca_matrix()
    assert lca.shape == (3, 3)
    assert lca[0, 0] == 0
    assert lca[0, 1] == 1
    assert lca[0, 2] == 2
    lca = doc[2:].get_lca_matrix()
    assert lca.shape == (2, 2)
    assert lca[0, 0] == 0
    assert lca[0, 1] == 1
    assert lca[1, 0] == 1
    assert lca[1, 1] == 1
    tokens = en_tokenizer('I like New York in Autumn')
    doc = Doc(tokens.vocab, words=[t.text for t in tokens], heads=[1, 1, 3, 1, 3, 4], deps=['dep'] * len(tokens))
    lca = doc[1:4].get_lca_matrix()
    assert_array_equal(lca, numpy.asarray([[0, 0, 0], [0, 1, 2], [0, 2, 2]]))

def test_span_similarity_match():
    if False:
        for i in range(10):
            print('nop')
    doc = Doc(Vocab(), words=['a', 'b', 'a', 'b'])
    span1 = doc[:2]
    span2 = doc[2:]
    with pytest.warns(UserWarning):
        assert span1.similarity(span2) == 1.0
        assert span1.similarity(doc) == 0.0
        assert span1[:1].similarity(doc.vocab['a']) == 1.0

def test_spans_default_sentiment(en_tokenizer):
    if False:
        while True:
            i = 10
    "Test span.sentiment property's default averaging behaviour"
    text = 'good stuff bad stuff'
    tokens = en_tokenizer(text)
    tokens.vocab[tokens[0].text].sentiment = 3.0
    tokens.vocab[tokens[2].text].sentiment = -2.0
    doc = Doc(tokens.vocab, words=[t.text for t in tokens])
    assert doc[:2].sentiment == 3.0 / 2
    assert doc[-2:].sentiment == -2.0 / 2
    assert doc[:-1].sentiment == (3.0 + -2) / 3.0

def test_spans_override_sentiment(en_tokenizer):
    if False:
        for i in range(10):
            print('nop')
    "Test span.sentiment property's default averaging behaviour"
    text = 'good stuff bad stuff'
    tokens = en_tokenizer(text)
    tokens.vocab[tokens[0].text].sentiment = 3.0
    tokens.vocab[tokens[2].text].sentiment = -2.0
    doc = Doc(tokens.vocab, words=[t.text for t in tokens])
    doc.user_span_hooks['sentiment'] = lambda span: 10.0
    assert doc[:2].sentiment == 10.0
    assert doc[-2:].sentiment == 10.0
    assert doc[:-1].sentiment == 10.0

def test_spans_are_hashable(en_tokenizer):
    if False:
        i = 10
        return i + 15
    'Test spans can be hashed.'
    text = 'good stuff bad stuff'
    tokens = en_tokenizer(text)
    span1 = tokens[:2]
    span2 = tokens[2:4]
    assert hash(span1) != hash(span2)
    span3 = tokens[0:2]
    assert hash(span3) == hash(span1)

def test_spans_by_character(doc):
    if False:
        return 10
    span1 = doc[1:-2]
    span2 = doc.char_span(span1.start_char, span1.end_char, label='GPE')
    assert span1.start_char == span2.start_char
    assert span1.end_char == span2.end_char
    assert span2.label_ == 'GPE'
    span2 = doc.char_span(span1.start_char, span1.end_char, label='GPE', alignment_mode='strict')
    assert span1.start_char == span2.start_char
    assert span1.end_char == span2.end_char
    assert span2.label_ == 'GPE'
    span2 = doc.char_span(span1.start_char - 3, span1.end_char, label='GPE', alignment_mode='contract')
    assert span1.start_char == span2.start_char
    assert span1.end_char == span2.end_char
    assert span2.label_ == 'GPE'
    span2 = doc.char_span(span1.start_char + 1, span1.end_char, label='GPE', alignment_mode='expand')
    assert span1.start_char == span2.start_char
    assert span1.end_char == span2.end_char
    assert span2.label_ == 'GPE'
    with pytest.raises(ValueError):
        span2 = doc.char_span(span1.start_char + 1, span1.end_char, label='GPE', alignment_mode='unk')
    span2 = doc[0:2].char_span(span1.start_char - 3, span1.end_char, label='GPE', alignment_mode='contract')
    assert span1.start_char == span2.start_char
    assert span1.end_char == span2.end_char
    assert span2.label_ == 'GPE'

def test_span_to_array(doc):
    if False:
        for i in range(10):
            print('nop')
    span = doc[1:-2]
    arr = span.to_array([ORTH, LENGTH])
    assert arr.shape == (len(span), 2)
    assert arr[0, 0] == span[0].orth
    assert arr[0, 1] == len(span[0])

def test_span_as_doc(doc):
    if False:
        print('Hello World!')
    span = doc[4:10]
    span_doc = span.as_doc()
    assert span.text == span_doc.text.strip()
    assert isinstance(span_doc, doc.__class__)
    assert span_doc is not doc
    assert span_doc[0].idx == 0
    assert len(span_doc.ents) == 0
    span_doc = doc[2:10].as_doc()
    assert len(span_doc.ents) == 1
    span_doc = doc[0:5].as_doc()
    assert len(span_doc.ents) == 0

@pytest.mark.usefixtures('clean_underscore')
def test_span_as_doc_user_data(doc):
    if False:
        return 10
    'Test that the user_data can be preserved (but not by default).'
    my_key = 'my_info'
    my_value = 342
    doc.user_data[my_key] = my_value
    Token.set_extension('is_x', default=False)
    doc[7]._.is_x = True
    span = doc[4:10]
    span_doc_with = span.as_doc(copy_user_data=True)
    span_doc_without = span.as_doc()
    assert doc.user_data.get(my_key, None) is my_value
    assert span_doc_with.user_data.get(my_key, None) is my_value
    assert span_doc_without.user_data.get(my_key, None) is None
    for i in range(len(span_doc_with)):
        if i != 3:
            assert span_doc_with[i]._.is_x is False
        else:
            assert span_doc_with[i]._.is_x is True
    assert not any([t._.is_x for t in span_doc_without])

def test_span_string_label_kb_id(doc):
    if False:
        print('Hello World!')
    span = Span(doc, 0, 1, label='hello', kb_id='Q342')
    assert span.label_ == 'hello'
    assert span.label == doc.vocab.strings['hello']
    assert span.kb_id_ == 'Q342'
    assert span.kb_id == doc.vocab.strings['Q342']

def test_span_string_label_id(doc):
    if False:
        print('Hello World!')
    span = Span(doc, 0, 1, label='hello', span_id='Q342')
    assert span.label_ == 'hello'
    assert span.label == doc.vocab.strings['hello']
    assert span.id_ == 'Q342'
    assert span.id == doc.vocab.strings['Q342']

def test_span_attrs_writable(doc):
    if False:
        while True:
            i = 10
    span = Span(doc, 0, 1)
    span.label_ = 'label'
    span.kb_id_ = 'kb_id'
    span.id_ = 'id'

def test_span_ents_property(doc):
    if False:
        while True:
            i = 10
    doc.ents = [(doc.vocab.strings['PRODUCT'], 0, 1), (doc.vocab.strings['PRODUCT'], 7, 8), (doc.vocab.strings['PRODUCT'], 11, 14)]
    assert len(list(doc.ents)) == 3
    sentences = list(doc.sents)
    assert len(sentences) == 3
    assert len(sentences[0].ents) == 1
    assert sentences[0].ents[0].text == 'This'
    assert sentences[0].ents[0].label_ == 'PRODUCT'
    assert sentences[0].ents[0].start == 0
    assert sentences[0].ents[0].end == 1
    assert len(sentences[1].ents) == 1
    assert sentences[1].ents[0].text == 'another'
    assert sentences[1].ents[0].label_ == 'PRODUCT'
    assert sentences[1].ents[0].start == 7
    assert sentences[1].ents[0].end == 8
    assert sentences[2].ents[0].text == 'a third.'
    assert sentences[2].ents[0].label_ == 'PRODUCT'
    assert sentences[2].ents[0].start == 11
    assert sentences[2].ents[0].end == 14

def test_filter_spans(doc):
    if False:
        print('Hello World!')
    spans = [doc[1:4], doc[6:8], doc[1:4], doc[10:14]]
    filtered = filter_spans(spans)
    assert len(filtered) == 3
    assert filtered[0].start == 1 and filtered[0].end == 4
    assert filtered[1].start == 6 and filtered[1].end == 8
    assert filtered[2].start == 10 and filtered[2].end == 14
    spans = [doc[1:4], doc[1:3], doc[5:10], doc[7:9], doc[1:4]]
    filtered = filter_spans(spans)
    assert len(filtered) == 2
    assert len(filtered[0]) == 3
    assert len(filtered[1]) == 5
    assert filtered[0].start == 1 and filtered[0].end == 4
    assert filtered[1].start == 5 and filtered[1].end == 10
    spans = [doc[1:4], doc[2:5], doc[5:10], doc[7:9], doc[1:4]]
    filtered = filter_spans(spans)
    assert len(filtered) == 2
    assert len(filtered[0]) == 3
    assert len(filtered[1]) == 5
    assert filtered[0].start == 1 and filtered[0].end == 4
    assert filtered[1].start == 5 and filtered[1].end == 10

def test_span_eq_hash(doc, doc_not_parsed):
    if False:
        print('Hello World!')
    assert doc[0:2] == doc[0:2]
    assert doc[0:2] != doc[1:3]
    assert doc[0:2] != doc_not_parsed[0:2]
    assert hash(doc[0:2]) == hash(doc[0:2])
    assert hash(doc[0:2]) != hash(doc[1:3])
    assert hash(doc[0:2]) != hash(doc_not_parsed[0:2])
    assert doc[0:len(doc)] != doc[len(doc):len(doc) + 1]

def test_span_boundaries(doc):
    if False:
        i = 10
        return i + 15
    start = 1
    end = 5
    span = doc[start:end]
    for i in range(start, end):
        assert span[i - start] == doc[i]
    with pytest.raises(IndexError):
        span[-5]
    with pytest.raises(IndexError):
        span[5]
    empty_span_0 = doc[0:0]
    assert empty_span_0.text == ''
    assert empty_span_0.start == 0
    assert empty_span_0.end == 0
    assert empty_span_0.start_char == 0
    assert empty_span_0.end_char == 0
    empty_span_1 = doc[1:1]
    assert empty_span_1.text == ''
    assert empty_span_1.start == 1
    assert empty_span_1.end == 1
    assert empty_span_1.start_char == empty_span_1.end_char
    oob_span_start = doc[-len(doc) - 1:-len(doc) - 10]
    assert oob_span_start.text == ''
    assert oob_span_start.start == 0
    assert oob_span_start.end == 0
    assert oob_span_start.start_char == 0
    assert oob_span_start.end_char == 0
    oob_span_end = doc[len(doc) + 1:len(doc) + 10]
    assert oob_span_end.text == ''
    assert oob_span_end.start == len(doc)
    assert oob_span_end.end == len(doc)
    assert oob_span_end.start_char == len(doc.text)
    assert oob_span_end.end_char == len(doc.text)

def test_span_lemma(doc):
    if False:
        for i in range(10):
            print('nop')
    sp = doc[1:5]
    assert len(sp.text.split(' ')) == len(sp.lemma_.split(' '))

def test_sent(en_tokenizer):
    if False:
        for i in range(10):
            print('nop')
    doc = en_tokenizer('Check span.sent raises error if doc is not sentencized.')
    span = doc[1:3]
    assert not span.doc.has_annotation('SENT_START')
    with pytest.raises(ValueError):
        span.sent

def test_span_with_vectors(doc):
    if False:
        for i in range(10):
            print('nop')
    ops = get_current_ops()
    prev_vectors = doc.vocab.vectors
    vectors = [('apple', ops.asarray([1, 2, 3])), ('orange', ops.asarray([-1, -2, -3])), ('And', ops.asarray([-1, -1, -1])), ('juice', ops.asarray([5, 5, 10])), ('pie', ops.asarray([7, 6.3, 8.9]))]
    add_vecs_to_vocab(doc.vocab, vectors)
    assert_array_equal(ops.to_numpy(doc[0:0].vector), numpy.zeros((3,)))
    assert_array_equal(ops.to_numpy(doc[0:4].vector), numpy.zeros((3,)))
    assert_array_equal(ops.to_numpy(doc[10:11].vector), [-1, -1, -1])
    doc.vocab.vectors = prev_vectors

def test_span_comparison(doc):
    if False:
        for i in range(10):
            print('nop')
    assert Span(doc, 0, 3) == Span(doc, 0, 3)
    assert Span(doc, 0, 3, 'LABEL') == Span(doc, 0, 3, 'LABEL')
    assert Span(doc, 0, 3, 'LABEL', kb_id='KB_ID') == Span(doc, 0, 3, 'LABEL', kb_id='KB_ID')
    assert Span(doc, 0, 3) != Span(doc, 0, 3, 'LABEL')
    assert Span(doc, 0, 3) != Span(doc, 0, 3, 'LABEL', kb_id='KB_ID')
    assert Span(doc, 0, 3, 'LABEL') != Span(doc, 0, 3, 'LABEL', kb_id='KB_ID')
    assert Span(doc, 0, 3) <= Span(doc, 0, 3) and Span(doc, 0, 3) >= Span(doc, 0, 3)
    assert Span(doc, 0, 3, 'LABEL') <= Span(doc, 0, 3, 'LABEL') and Span(doc, 0, 3, 'LABEL') >= Span(doc, 0, 3, 'LABEL')
    assert Span(doc, 0, 3, 'LABEL', kb_id='KB_ID') <= Span(doc, 0, 3, 'LABEL', kb_id='KB_ID')
    assert Span(doc, 0, 3, 'LABEL', kb_id='KB_ID') >= Span(doc, 0, 3, 'LABEL', kb_id='KB_ID')
    assert Span(doc, 0, 3) < Span(doc, 0, 3, '', kb_id='KB_ID') < Span(doc, 0, 3, 'LABEL') < Span(doc, 0, 3, 'LABEL', kb_id='KB_ID')
    assert Span(doc, 0, 3) <= Span(doc, 0, 3, '', kb_id='KB_ID') <= Span(doc, 0, 3, 'LABEL') <= Span(doc, 0, 3, 'LABEL', kb_id='KB_ID')
    assert Span(doc, 0, 3, 'LABEL', kb_id='KB_ID') > Span(doc, 0, 3, 'LABEL') > Span(doc, 0, 3, '', kb_id='KB_ID') > Span(doc, 0, 3)
    assert Span(doc, 0, 3, 'LABEL', kb_id='KB_ID') >= Span(doc, 0, 3, 'LABEL') >= Span(doc, 0, 3, '', kb_id='KB_ID') >= Span(doc, 0, 3)
    assert Span(doc, 0, 3, 'LABEL', kb_id='KB_ID') < Span(doc, 0, 4, 'LABEL', kb_id='KB_ID')
    assert Span(doc, 0, 3, 'LABEL', kb_id='KB_ID') < Span(doc, 0, 4)
    assert Span(doc, 0, 3, 'LABEL', kb_id='KB_ID') <= Span(doc, 0, 4)
    assert Span(doc, 0, 4) > Span(doc, 0, 3, 'LABEL', kb_id='KB_ID')
    assert Span(doc, 0, 4) >= Span(doc, 0, 3, 'LABEL', kb_id='KB_ID')
    assert Span(doc, 0, 3, 'LABEL', kb_id='KB_ID') != Span(doc, 1, 3, 'LABEL', kb_id='KB_ID')
    assert Span(doc, 0, 3, 'LABEL', kb_id='KB_ID') < Span(doc, 1, 3)
    assert Span(doc, 0, 3, 'LABEL', kb_id='KB_ID') <= Span(doc, 1, 3)
    assert Span(doc, 1, 3) > Span(doc, 0, 3, 'LABEL', kb_id='KB_ID')
    assert Span(doc, 1, 3) >= Span(doc, 0, 3, 'LABEL', kb_id='KB_ID')
    assert Span(doc, 0, 4, 'LABEL', kb_id='KB_ID') != Span(doc, 1, 3, 'LABEL', kb_id='KB_ID')
    assert Span(doc, 0, 4, 'LABEL', kb_id='KB_ID') < Span(doc, 1, 3)
    assert Span(doc, 0, 4, 'LABEL', kb_id='KB_ID') <= Span(doc, 1, 3)
    assert Span(doc, 1, 3) > Span(doc, 0, 4, 'LABEL', kb_id='KB_ID')
    assert Span(doc, 1, 3) >= Span(doc, 0, 4, 'LABEL', kb_id='KB_ID')
    assert Span(doc, 1, 3, span_id='AAA') < Span(doc, 1, 3, span_id='BBB')

@pytest.mark.parametrize('start,end,expected_sentences,expected_sentences_with_hook', [(0, 14, 3, 7), (3, 6, 2, 2), (0, 4, 1, 2), (0, 3, 1, 2), (9, 14, 2, 3), (10, 14, 1, 2), (11, 14, 1, 2), (0, 0, 1, 1)])
def test_span_sents(doc, start, end, expected_sentences, expected_sentences_with_hook):
    if False:
        return 10
    assert len(list(doc[start:end].sents)) == expected_sentences

    def user_hook(doc):
        if False:
            return 10
        return [doc[ii:ii + 2] for ii in range(0, len(doc), 2)]
    doc.user_hooks['sents'] = user_hook
    assert len(list(doc[start:end].sents)) == expected_sentences_with_hook
    doc.user_span_hooks['sents'] = lambda x: [x]
    assert list(doc[start:end].sents)[0] == doc[start:end]
    assert len(list(doc[start:end].sents)) == 1

def test_span_sents_not_parsed(doc_not_parsed):
    if False:
        while True:
            i = 10
    with pytest.raises(ValueError):
        list(Span(doc_not_parsed, 0, 3).sents)

def test_span_group_copy(doc):
    if False:
        return 10
    doc.spans['test'] = [doc[0:1], doc[2:4]]
    assert len(doc.spans['test']) == 2
    doc_copy = doc.copy()
    assert len(doc_copy.spans['test']) == 2
    doc.spans['test'].append(doc[3:4])
    assert len(doc.spans['test']) == 3
    assert len(doc_copy.spans['test']) == 2

def test_for_partial_ent_sents():
    if False:
        for i in range(10):
            print('nop')
    'Spans may be associated with multiple sentences. These .sents should always be complete, not partial, sentences,\n    which this tests for.\n    '
    doc = Doc(English().vocab, words=["Mahler's", 'Symphony', 'No.', '8', 'was', 'beautiful.'], sent_starts=[1, 0, 0, 1, 0, 0])
    doc.set_ents([Span(doc, 1, 4, 'WORK')])
    for (doc_sent, ent_sent) in zip(doc.sents, doc.ents[0].sents):
        assert doc_sent == ent_sent

def test_for_no_ent_sents():
    if False:
        for i in range(10):
            print('nop')
    "Span.sents() should set .sents correctly, even if Span in question is trailing and doesn't form a full\n    sentence.\n    "
    doc = Doc(English().vocab, words=['This', 'is', 'a', 'test.', 'ENTITY'], sent_starts=[1, 0, 0, 0, 1])
    doc.set_ents([Span(doc, 4, 5, 'WORK')])
    sents = list(doc.ents[0].sents)
    assert len(sents) == 1
    assert str(sents[0]) == str(doc.ents[0].sent) == 'ENTITY'

def test_span_api_richcmp_other(en_tokenizer):
    if False:
        for i in range(10):
            print('nop')
    doc1 = en_tokenizer('a b')
    doc2 = en_tokenizer('b c')
    assert not doc1[1:2] == doc1[1]
    assert not doc1[1:2] == doc2[0]
    assert not doc1[1:2] == doc2[0:1]
    assert not doc1[0:1] == doc2