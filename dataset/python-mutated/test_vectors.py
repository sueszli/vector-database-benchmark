import numpy
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
from thinc.api import NumpyOps, get_current_ops
from spacy.lang.en import English
from spacy.strings import hash_string
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc
from spacy.training.initialize import convert_vectors
from spacy.vectors import Vectors
from spacy.vocab import Vocab
from ..util import add_vecs_to_vocab, get_cosine, make_tempdir
OPS = get_current_ops()

@pytest.fixture
def strings():
    if False:
        return 10
    return ['apple', 'orange']

@pytest.fixture
def vectors():
    if False:
        for i in range(10):
            print('nop')
    return [('apple', OPS.asarray([1, 2, 3])), ('orange', OPS.asarray([-1, -2, -3])), ('and', OPS.asarray([-1, -1, -1])), ('juice', OPS.asarray([5, 5, 10])), ('pie', OPS.asarray([7, 6.3, 8.9]))]

@pytest.fixture
def data():
    if False:
        return 10
    return numpy.asarray([[0.0, 1.0, 2.0], [3.0, -2.0, 4.0]], dtype='f')

@pytest.fixture
def most_similar_vectors_data():
    if False:
        for i in range(10):
            print('nop')
    return numpy.asarray([[0.0, 1.0, 2.0], [1.0, -2.0, 4.0], [1.0, 1.0, -1.0], [2.0, 3.0, 1.0]], dtype='f')

@pytest.fixture
def most_similar_vectors_keys():
    if False:
        return 10
    return ['a', 'b', 'c', 'd']

@pytest.fixture
def resize_data():
    if False:
        while True:
            i = 10
    return numpy.asarray([[0.0, 1.0], [2.0, 3.0]], dtype='f')

@pytest.fixture()
def vocab(en_vocab, vectors):
    if False:
        print('Hello World!')
    add_vecs_to_vocab(en_vocab, vectors)
    return en_vocab

@pytest.fixture()
def tokenizer_v(vocab):
    if False:
        return 10
    return Tokenizer(vocab, {}, None, None, None)

@pytest.mark.issue(1518)
def test_issue1518():
    if False:
        for i in range(10):
            print('nop')
    'Test vectors.resize() works.'
    vectors = Vectors(shape=(10, 10))
    vectors.add('hello', row=2)
    vectors.resize((5, 9))

@pytest.mark.issue(1539)
def test_issue1539():
    if False:
        for i in range(10):
            print('nop')
    "Ensure vectors.resize() doesn't try to modify dictionary during iteration."
    v = Vectors(shape=(10, 10), keys=[5, 3, 98, 100])
    v.resize((100, 100))

@pytest.mark.issue(1807)
def test_issue1807():
    if False:
        return 10
    'Test vocab.set_vector also adds the word to the vocab.'
    vocab = Vocab(vectors_name='test_issue1807')
    assert 'hello' not in vocab
    vocab.set_vector('hello', numpy.ones((50,), dtype='f'))
    assert 'hello' in vocab

@pytest.mark.issue(2871)
def test_issue2871():
    if False:
        while True:
            i = 10
    'Test that vectors recover the correct key for spaCy reserved words.'
    words = ['dog', 'cat', 'SUFFIX']
    vocab = Vocab(vectors_name='test_issue2871')
    vocab.vectors.resize(shape=(3, 10))
    vector_data = numpy.zeros((3, 10), dtype='f')
    for word in words:
        _ = vocab[word]
        vocab.set_vector(word, vector_data[0])
    vocab.vectors.name = 'dummy_vectors'
    assert vocab['dog'].rank == 0
    assert vocab['cat'].rank == 1
    assert vocab['SUFFIX'].rank == 2
    assert vocab.vectors.find(key='dog') == 0
    assert vocab.vectors.find(key='cat') == 1
    assert vocab.vectors.find(key='SUFFIX') == 2

@pytest.mark.issue(3412)
def test_issue3412():
    if False:
        return 10
    data = numpy.asarray([[0, 0, 0], [1, 2, 3], [9, 8, 7]], dtype='f')
    vectors = Vectors(data=data, keys=['A', 'B', 'C'])
    (keys, best_rows, scores) = vectors.most_similar(numpy.asarray([[9, 8, 7], [0, 0, 0]], dtype='f'))
    assert best_rows[0] == 2

@pytest.mark.issue(4725)
def test_issue4725_2():
    if False:
        i = 10
        return i + 15
    if isinstance(get_current_ops, NumpyOps):
        vocab = Vocab(vectors_name='test_vocab_add_vector')
        data = numpy.ndarray((5, 3), dtype='f')
        data[0] = 1.0
        data[1] = 2.0
        vocab.set_vector('cat', data[0])
        vocab.set_vector('dog', data[1])
        nlp = English(vocab=vocab)
        nlp.add_pipe('ner')
        nlp.initialize()
        docs = ['Kurt is in London.'] * 10
        for _ in nlp.pipe(docs, batch_size=2, n_process=2):
            pass

def test_init_vectors_with_resize_shape(strings, resize_data):
    if False:
        i = 10
        return i + 15
    v = Vectors(shape=(len(strings), 3))
    v.resize(shape=resize_data.shape)
    assert v.shape == resize_data.shape
    assert v.shape != (len(strings), 3)

def test_init_vectors_with_resize_data(data, resize_data):
    if False:
        return 10
    v = Vectors(data=data)
    v.resize(shape=resize_data.shape)
    assert v.shape == resize_data.shape
    assert v.shape != data.shape

def test_get_vector_resize(strings, data):
    if False:
        print('Hello World!')
    strings = [hash_string(s) for s in strings]
    v = Vectors(data=data)
    resized_dim = v.shape[1] - 1
    v.resize(shape=(v.shape[0], resized_dim))
    for (i, string) in enumerate(strings):
        v.add(string, row=i)
    assert list(v[strings[0]]) == list(data[0, :resized_dim])
    assert list(v[strings[1]]) == list(data[1, :resized_dim])
    v = Vectors(data=data)
    resized_dim = v.shape[1] + 1
    v.resize(shape=(v.shape[0], resized_dim))
    for (i, string) in enumerate(strings):
        v.add(string, row=i)
    assert list(v[strings[0]]) == list(data[0]) + [0]
    assert list(v[strings[1]]) == list(data[1]) + [0]

def test_init_vectors_with_data(strings, data):
    if False:
        while True:
            i = 10
    v = Vectors(data=data)
    assert v.shape == data.shape

def test_init_vectors_with_shape(strings):
    if False:
        print('Hello World!')
    v = Vectors(shape=(len(strings), 3))
    assert v.shape == (len(strings), 3)
    assert v.is_full is False

def test_get_vector(strings, data):
    if False:
        print('Hello World!')
    v = Vectors(data=data)
    strings = [hash_string(s) for s in strings]
    for (i, string) in enumerate(strings):
        v.add(string, row=i)
    assert list(v[strings[0]]) == list(data[0])
    assert list(v[strings[0]]) != list(data[1])
    assert list(v[strings[1]]) != list(data[0])

def test_set_vector(strings, data):
    if False:
        for i in range(10):
            print('nop')
    orig = data.copy()
    v = Vectors(data=data)
    strings = [hash_string(s) for s in strings]
    for (i, string) in enumerate(strings):
        v.add(string, row=i)
    assert list(v[strings[0]]) == list(orig[0])
    assert list(v[strings[0]]) != list(orig[1])
    v[strings[0]] = data[1]
    assert list(v[strings[0]]) == list(orig[1])
    assert list(v[strings[0]]) != list(orig[0])

def test_vectors_most_similar(most_similar_vectors_data, most_similar_vectors_keys):
    if False:
        while True:
            i = 10
    v = Vectors(data=most_similar_vectors_data, keys=most_similar_vectors_keys)
    (_, best_rows, _) = v.most_similar(v.data, batch_size=2, n=2, sort=True)
    assert all((row[0] == i for (i, row) in enumerate(best_rows)))
    with pytest.raises(ValueError):
        v.most_similar(v.data, batch_size=2, n=10, sort=True)

def test_vectors_most_similar_identical():
    if False:
        return 10
    'Test that most similar identical vectors are assigned a score of 1.0.'
    data = numpy.asarray([[4, 2, 2, 2], [4, 2, 2, 2], [1, 1, 1, 1]], dtype='f')
    v = Vectors(data=data, keys=['A', 'B', 'C'])
    (keys, _, scores) = v.most_similar(numpy.asarray([[4, 2, 2, 2]], dtype='f'))
    assert scores[0][0] == 1.0
    data = numpy.asarray([[1, 2, 3], [1, 2, 3], [1, 1, 1]], dtype='f')
    v = Vectors(data=data, keys=['A', 'B', 'C'])
    (keys, _, scores) = v.most_similar(numpy.asarray([[1, 2, 3]], dtype='f'))
    assert scores[0][0] == 1.0

@pytest.mark.parametrize('text', ['apple and orange'])
def test_vectors_token_vector(tokenizer_v, vectors, text):
    if False:
        i = 10
        return i + 15
    doc = tokenizer_v(text)
    assert vectors[0][0] == doc[0].text
    assert all([a == b for (a, b) in zip(vectors[0][1], doc[0].vector)])
    assert vectors[1][0] == doc[2].text
    assert all([a == b for (a, b) in zip(vectors[1][1], doc[2].vector)])

@pytest.mark.parametrize('text', ['apple', 'orange'])
def test_vectors_lexeme_vector(vocab, text):
    if False:
        return 10
    lex = vocab[text]
    assert list(lex.vector)
    assert lex.vector_norm

@pytest.mark.parametrize('text', [['apple', 'and', 'orange']])
def test_vectors_doc_vector(vocab, text):
    if False:
        while True:
            i = 10
    doc = Doc(vocab, words=text)
    assert list(doc.vector)
    assert doc.vector_norm

@pytest.mark.parametrize('text', [['apple', 'and', 'orange']])
def test_vectors_span_vector(vocab, text):
    if False:
        for i in range(10):
            print('nop')
    span = Doc(vocab, words=text)[0:2]
    assert list(span.vector)
    assert span.vector_norm

@pytest.mark.parametrize('text', ['apple orange'])
def test_vectors_token_token_similarity(tokenizer_v, text):
    if False:
        return 10
    doc = tokenizer_v(text)
    assert doc[0].similarity(doc[1]) == doc[1].similarity(doc[0])
    assert -1.0 < doc[0].similarity(doc[1]) < 1.0

@pytest.mark.parametrize('text1,text2', [('apple', 'orange')])
def test_vectors_token_lexeme_similarity(tokenizer_v, vocab, text1, text2):
    if False:
        for i in range(10):
            print('nop')
    token = tokenizer_v(text1)
    lex = vocab[text2]
    assert token.similarity(lex) == lex.similarity(token)
    assert -1.0 < token.similarity(lex) < 1.0

@pytest.mark.parametrize('text', [['apple', 'orange', 'juice']])
def test_vectors_token_span_similarity(vocab, text):
    if False:
        print('Hello World!')
    doc = Doc(vocab, words=text)
    assert doc[0].similarity(doc[1:3]) == doc[1:3].similarity(doc[0])
    assert -1.0 < doc[0].similarity(doc[1:3]) < 1.0

@pytest.mark.parametrize('text', [['apple', 'orange', 'juice']])
def test_vectors_token_doc_similarity(vocab, text):
    if False:
        while True:
            i = 10
    doc = Doc(vocab, words=text)
    assert doc[0].similarity(doc) == doc.similarity(doc[0])
    assert -1.0 < doc[0].similarity(doc) < 1.0

@pytest.mark.parametrize('text', [['apple', 'orange', 'juice']])
def test_vectors_lexeme_span_similarity(vocab, text):
    if False:
        return 10
    doc = Doc(vocab, words=text)
    lex = vocab[text[0]]
    assert lex.similarity(doc[1:3]) == doc[1:3].similarity(lex)
    assert -1.0 < doc.similarity(doc[1:3]) < 1.0

@pytest.mark.parametrize('text1,text2', [('apple', 'orange')])
def test_vectors_lexeme_lexeme_similarity(vocab, text1, text2):
    if False:
        while True:
            i = 10
    lex1 = vocab[text1]
    lex2 = vocab[text2]
    assert lex1.similarity(lex2) == lex2.similarity(lex1)
    assert -1.0 < lex1.similarity(lex2) < 1.0

@pytest.mark.parametrize('text', [['apple', 'orange', 'juice']])
def test_vectors_lexeme_doc_similarity(vocab, text):
    if False:
        i = 10
        return i + 15
    doc = Doc(vocab, words=text)
    lex = vocab[text[0]]
    assert lex.similarity(doc) == doc.similarity(lex)
    assert -1.0 < lex.similarity(doc) < 1.0

@pytest.mark.parametrize('text', [['apple', 'orange', 'juice']])
def test_vectors_span_span_similarity(vocab, text):
    if False:
        print('Hello World!')
    doc = Doc(vocab, words=text)
    assert doc[0:2].similarity(doc[1:3]) == doc[1:3].similarity(doc[0:2])
    assert -1.0 < doc[0:2].similarity(doc[1:3]) < 1.0

@pytest.mark.parametrize('text', [['apple', 'orange', 'juice']])
def test_vectors_span_doc_similarity(vocab, text):
    if False:
        while True:
            i = 10
    doc = Doc(vocab, words=text)
    assert doc[0:2].similarity(doc) == doc.similarity(doc[0:2])
    assert -1.0 < doc[0:2].similarity(doc) < 1.0

@pytest.mark.parametrize('text1,text2', [(['apple', 'and', 'apple', 'pie'], ['orange', 'juice'])])
def test_vectors_doc_doc_similarity(vocab, text1, text2):
    if False:
        i = 10
        return i + 15
    doc1 = Doc(vocab, words=text1)
    doc2 = Doc(vocab, words=text2)
    assert doc1.similarity(doc2) == doc2.similarity(doc1)
    assert -1.0 < doc1.similarity(doc2) < 1.0

def test_vocab_add_vector():
    if False:
        i = 10
        return i + 15
    vocab = Vocab(vectors_name='test_vocab_add_vector')
    data = OPS.xp.ndarray((5, 3), dtype='f')
    data[0] = 1.0
    data[1] = 2.0
    vocab.set_vector('cat', data[0])
    vocab.set_vector('dog', data[1])
    cat = vocab['cat']
    assert list(cat.vector) == [1.0, 1.0, 1.0]
    dog = vocab['dog']
    assert list(dog.vector) == [2.0, 2.0, 2.0]
    with pytest.raises(ValueError):
        vocab.vectors.add(vocab['hamster'].orth, row=1000000)

def test_vocab_prune_vectors():
    if False:
        return 10
    vocab = Vocab(vectors_name='test_vocab_prune_vectors')
    _ = vocab['cat']
    _ = vocab['dog']
    _ = vocab['kitten']
    data = OPS.xp.ndarray((5, 3), dtype='f')
    data[0] = OPS.asarray([1.0, 1.2, 1.1])
    data[1] = OPS.asarray([0.3, 1.3, 1.0])
    data[2] = OPS.asarray([0.9, 1.22, 1.05])
    vocab.set_vector('cat', data[0])
    vocab.set_vector('dog', data[1])
    vocab.set_vector('kitten', data[2])
    remap = vocab.prune_vectors(2, batch_size=2)
    assert list(remap.keys()) == ['kitten']
    (neighbour, similarity) = list(remap.values())[0]
    assert neighbour == 'cat', remap
    cosine = get_cosine(data[0], data[2])
    assert_allclose(float(similarity), cosine, atol=0.0001, rtol=0.001)

def test_vectors_serialize():
    if False:
        for i in range(10):
            print('nop')
    data = OPS.asarray([[4, 2, 2, 2], [4, 2, 2, 2], [1, 1, 1, 1]], dtype='f')
    v = Vectors(data=data, keys=['A', 'B', 'C'])
    b = v.to_bytes()
    v_r = Vectors()
    v_r.from_bytes(b)
    assert_equal(OPS.to_numpy(v.data), OPS.to_numpy(v_r.data))
    assert v.key2row == v_r.key2row
    v.resize((5, 4))
    v_r.resize((5, 4))
    row = v.add('D', vector=OPS.asarray([1, 2, 3, 4], dtype='f'))
    row_r = v_r.add('D', vector=OPS.asarray([1, 2, 3, 4], dtype='f'))
    assert row == row_r
    assert_equal(OPS.to_numpy(v.data), OPS.to_numpy(v_r.data))
    assert v.is_full == v_r.is_full
    with make_tempdir() as d:
        v.to_disk(d)
        v_r.from_disk(d)
        assert_equal(OPS.to_numpy(v.data), OPS.to_numpy(v_r.data))
        assert v.key2row == v_r.key2row
        v.resize((5, 4))
        v_r.resize((5, 4))
        row = v.add('D', vector=OPS.asarray([10, 20, 30, 40], dtype='f'))
        row_r = v_r.add('D', vector=OPS.asarray([10, 20, 30, 40], dtype='f'))
        assert row == row_r
        assert_equal(OPS.to_numpy(v.data), OPS.to_numpy(v_r.data))
        assert v.attr == v_r.attr

def test_vector_is_oov():
    if False:
        i = 10
        return i + 15
    vocab = Vocab(vectors_name='test_vocab_is_oov')
    data = OPS.xp.ndarray((5, 3), dtype='f')
    data[0] = 1.0
    data[1] = 2.0
    vocab.set_vector('cat', data[0])
    vocab.set_vector('dog', data[1])
    assert vocab['cat'].is_oov is False
    assert vocab['dog'].is_oov is False
    assert vocab['hamster'].is_oov is True

def test_init_vectors_unset():
    if False:
        for i in range(10):
            print('nop')
    v = Vectors(shape=(10, 10))
    assert v.is_full is False
    assert v.shape == (10, 10)
    with pytest.raises(ValueError):
        v = Vectors(shape=(10, 10), mode='floret')
    v = Vectors(data=OPS.xp.zeros((10, 10)), mode='floret', hash_count=1)
    assert v.is_full is True

def test_vectors_clear():
    if False:
        while True:
            i = 10
    data = OPS.asarray([[4, 2, 2, 2], [4, 2, 2, 2], [1, 1, 1, 1]], dtype='f')
    v = Vectors(data=data, keys=['A', 'B', 'C'])
    assert v.is_full is True
    assert hash_string('A') in v
    v.clear()
    assert v.key2row == {}
    assert list(v) == []
    assert v.is_full is False
    assert 'A' not in v
    with pytest.raises(KeyError):
        v['A']

def test_vectors_get_batch():
    if False:
        while True:
            i = 10
    data = OPS.asarray([[4, 2, 2, 2], [4, 2, 2, 2], [1, 1, 1, 1]], dtype='f')
    v = Vectors(data=data, keys=['A', 'B', 'C'])
    words = ['C', 'B', 'A', v.strings['B']]
    rows = v.find(keys=words)
    vecs = OPS.as_contig(v.data[rows])
    assert_equal(OPS.to_numpy(vecs), OPS.to_numpy(v.get_batch(words)))

def test_vectors_deduplicate():
    if False:
        for i in range(10):
            print('nop')
    data = OPS.asarray([[1, 1], [2, 2], [3, 4], [1, 1], [3, 4]], dtype='f')
    v = Vectors(data=data, keys=['a1', 'b1', 'c1', 'a2', 'c2'])
    vocab = Vocab()
    vocab.vectors = v
    assert vocab.vectors.key2row[v.strings['a1']] != vocab.vectors.key2row[v.strings['a2']]
    assert vocab.vectors.key2row[v.strings['c1']] != vocab.vectors.key2row[v.strings['c2']]
    vocab.deduplicate_vectors()
    assert vocab.vectors.shape[0] == 3
    assert_equal(numpy.unique(OPS.to_numpy(vocab.vectors.data), axis=0), OPS.to_numpy(vocab.vectors.data))
    assert vocab.vectors.key2row[v.strings['a1']] == vocab.vectors.key2row[v.strings['a2']]
    assert vocab.vectors.key2row[v.strings['c1']] == vocab.vectors.key2row[v.strings['c2']]
    vocab_b = vocab.to_bytes()
    vocab.deduplicate_vectors()
    assert vocab_b == vocab.to_bytes()

@pytest.fixture()
def floret_vectors_hashvec_str():
    if False:
        print('Hello World!')
    'The full hashvec table from floret with the settings:\n    bucket 10, dim 10, minn 2, maxn 3, hash count 2, hash seed 2166136261,\n    bow <, eow >'
    return '10 10 2 3 2 2166136261 < >\n0 -2.2611 3.9302 2.6676 -11.233 0.093715 -10.52 -9.6463 -0.11853 2.101 -0.10145\n1 -3.12 -1.7981 10.7 -6.171 4.4527 10.967 9.073 6.2056 -6.1199 -2.0402\n2 9.5689 5.6721 -8.4832 -1.2249 2.1871 -3.0264 -2.391 -5.3308 -3.2847 -4.0382\n3 3.6268 4.2759 -1.7007 1.5002 5.5266 1.8716 -12.063 0.26314 2.7645 2.4929\n4 -11.683 -7.7068 2.1102 2.214 7.2202 0.69799 3.2173 -5.382 -2.0838 5.0314\n5 -4.3024 8.0241 2.0714 -1.0174 -0.28369 1.7622 7.8797 -1.7795 6.7541 5.6703\n6 8.3574 -5.225 8.6529 8.5605 -8.9465 3.767 -5.4636 -1.4635 -0.98947 -0.58025\n7 -10.01 3.3894 -4.4487 1.1669 -11.904 6.5158 4.3681 0.79913 -6.9131 -8.687\n8 -5.4576 7.1019 -8.8259 1.7189 4.955 -8.9157 -3.8905 -0.60086 -2.1233 5.892\n9 8.0678 -4.4142 3.6236 4.5889 -2.7611 2.4455 0.67096 -4.2822 2.0875 4.6274\n'

@pytest.fixture()
def floret_vectors_vec_str():
    if False:
        for i in range(10):
            print('nop')
    'The top 10 rows from floret with the settings above, to verify\n    that the spacy floret vectors are equivalent to the fasttext static\n    vectors.'
    return '10 10\n, -5.7814 2.6918 0.57029 -3.6985 -2.7079 1.4406 1.0084 1.7463 -3.8625 -3.0565\n. 3.8016 -1.759 0.59118 3.3044 -0.72975 0.45221 -2.1412 -3.8933 -2.1238 -0.47409\nder 0.08224 2.6601 -1.173 1.1549 -0.42821 -0.097268 -2.5589 -1.609 -0.16968 0.84687\ndie -2.8781 0.082576 1.9286 -0.33279 0.79488 3.36 3.5609 -0.64328 -2.4152 0.17266\nund 2.1558 1.8606 -1.382 0.45424 -0.65889 1.2706 0.5929 -2.0592 -2.6949 -1.6015\n" -1.1242 1.4588 -1.6263 1.0382 -2.7609 -0.99794 -0.83478 -1.5711 -1.2137 1.0239\nin -0.87635 2.0958 4.0018 -2.2473 -1.2429 2.3474 1.8846 0.46521 -0.506 -0.26653\nvon -0.10589 1.196 1.1143 -0.40907 -1.0848 -0.054756 -2.5016 -1.0381 -0.41598 0.36982\n( 0.59263 2.1856 0.67346 1.0769 1.0701 1.2151 1.718 -3.0441 2.7291 3.719\n) 0.13812 3.3267 1.657 0.34729 -3.5459 0.72372 0.63034 -1.6145 1.2733 0.37798\n'

def test_floret_vectors(floret_vectors_vec_str, floret_vectors_hashvec_str):
    if False:
        return 10
    nlp = English()
    nlp_plain = English()
    with make_tempdir() as tmpdir:
        p = tmpdir / 'test.hashvec'
        with open(p, 'w') as fileh:
            fileh.write(floret_vectors_hashvec_str)
        convert_vectors(nlp, p, truncate=0, prune=-1, mode='floret')
        p = tmpdir / 'test.vec'
        with open(p, 'w') as fileh:
            fileh.write(floret_vectors_vec_str)
        convert_vectors(nlp_plain, p, truncate=0, prune=-1)
    word = 'der'
    ngrams = nlp.vocab.vectors._get_ngrams(word)
    assert ngrams == ['<der>', '<d', 'de', 'er', 'r>', '<de', 'der', 'er>']
    rows = OPS.xp.asarray([h % nlp.vocab.vectors.shape[0] for ngram in ngrams for h in nlp.vocab.vectors._get_ngram_hashes(ngram)], dtype='uint32')
    assert_equal(OPS.to_numpy(rows), numpy.asarray([5, 6, 7, 5, 8, 2, 8, 9, 3, 3, 4, 6, 7, 3, 0, 2]))
    assert len(rows) == len(ngrams) * nlp.vocab.vectors.hash_count
    for word in nlp_plain.vocab.vectors:
        word = nlp_plain.vocab.strings.as_string(word)
        assert_almost_equal(nlp.vocab[word].vector, nlp_plain.vocab[word].vector, decimal=3)
        assert nlp.vocab[word * 5].has_vector
    assert nlp_plain.vocab.vectors.n_keys > 0
    assert nlp.vocab.vectors.n_keys == -1
    words = [s for s in nlp_plain.vocab.vectors]
    single_vecs = OPS.to_numpy(OPS.asarray([nlp.vocab[word].vector for word in words]))
    batch_vecs = OPS.to_numpy(nlp.vocab.vectors.get_batch(words))
    assert_equal(single_vecs, batch_vecs)
    assert_equal(OPS.to_numpy(nlp.vocab[''].vector), numpy.zeros((nlp.vocab.vectors.shape[0],)))
    assert_equal(OPS.to_numpy(nlp.vocab.vectors.get_batch([''])), numpy.zeros((1, nlp.vocab.vectors.shape[0])))
    assert_equal(OPS.to_numpy(nlp.vocab.vectors.get_batch(['a', '', 'b'])[1]), numpy.zeros((nlp.vocab.vectors.shape[0],)))
    vector = list(range(nlp.vocab.vectors.shape[1]))
    orig_bytes = nlp.vocab.vectors.to_bytes(exclude=['strings'])
    with pytest.warns(UserWarning):
        nlp.vocab.set_vector('the', vector)
    assert orig_bytes == nlp.vocab.vectors.to_bytes(exclude=['strings'])
    with pytest.warns(UserWarning):
        nlp.vocab[word].vector = vector
    assert orig_bytes == nlp.vocab.vectors.to_bytes(exclude=['strings'])
    with pytest.warns(UserWarning):
        nlp.vocab.vectors.add('the', row=6)
    assert orig_bytes == nlp.vocab.vectors.to_bytes(exclude=['strings'])
    with pytest.warns(UserWarning):
        nlp.vocab.vectors.resize(shape=(100, 10))
    assert orig_bytes == nlp.vocab.vectors.to_bytes(exclude=['strings'])
    with pytest.raises(ValueError):
        nlp.vocab.vectors.clear()
    with make_tempdir() as d:
        nlp.vocab.to_disk(d)
        vocab_r = Vocab()
        vocab_r.from_disk(d)
        assert nlp.vocab.vectors.to_bytes() == vocab_r.vectors.to_bytes()
        assert_equal(OPS.to_numpy(nlp.vocab.vectors.data), OPS.to_numpy(vocab_r.vectors.data))
        assert_equal(nlp.vocab.vectors._get_cfg(), vocab_r.vectors._get_cfg())
        assert_almost_equal(OPS.to_numpy(nlp.vocab[word].vector), OPS.to_numpy(vocab_r[word].vector), decimal=6)

def test_equality():
    if False:
        for i in range(10):
            print('nop')
    vectors1 = Vectors(shape=(10, 10))
    vectors2 = Vectors(shape=(10, 8))
    assert vectors1 != vectors2
    vectors2 = Vectors(shape=(10, 10))
    assert vectors1 == vectors2
    vectors1.add('hello', row=2)
    assert vectors1 != vectors2
    vectors2.add('hello', row=2)
    assert vectors1 == vectors2
    vectors1.resize((5, 9))
    vectors2.resize((5, 9))
    assert vectors1 == vectors2

def test_vectors_attr():
    if False:
        return 10
    data = numpy.asarray([[0, 0, 0], [1, 2, 3], [9, 8, 7]], dtype='f')
    nlp = English()
    nlp.vocab.vectors = Vectors(data=data, keys=['A', 'B', 'C'])
    assert nlp.vocab.strings['A'] in nlp.vocab.vectors.key2row
    assert nlp.vocab.strings['a'] not in nlp.vocab.vectors.key2row
    assert nlp.vocab['A'].has_vector is True
    assert nlp.vocab['a'].has_vector is False
    assert nlp('A')[0].has_vector is True
    assert nlp('a')[0].has_vector is False
    nlp = English()
    nlp.vocab.vectors = Vectors(data=data, keys=['a', 'b', 'c'], attr='LOWER')
    assert nlp.vocab.strings['A'] not in nlp.vocab.vectors.key2row
    assert nlp.vocab.strings['a'] in nlp.vocab.vectors.key2row
    assert nlp.vocab['A'].has_vector is True
    assert nlp.vocab['a'].has_vector is True
    assert nlp('A')[0].has_vector is True
    assert nlp('a')[0].has_vector is True
    assert nlp.vocab['D'].has_vector is False
    assert nlp.vocab['d'].has_vector is False
    nlp.vocab.set_vector('D', numpy.asarray([4, 5, 6]))
    assert nlp.vocab['D'].has_vector is True
    assert nlp.vocab['d'].has_vector is True