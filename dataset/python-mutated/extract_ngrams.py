from thinc.api import Model
from ..attrs import LOWER
from ..util import registry

@registry.layers('spacy.extract_ngrams.v1')
def extract_ngrams(ngram_size: int, attr: int=LOWER) -> Model:
    if False:
        for i in range(10):
            print('nop')
    model: Model = Model('extract_ngrams', forward)
    model.attrs['ngram_size'] = ngram_size
    model.attrs['attr'] = attr
    return model

def forward(model: Model, docs, is_train: bool):
    if False:
        i = 10
        return i + 15
    batch_keys = []
    batch_vals = []
    for doc in docs:
        unigrams = model.ops.asarray(doc.to_array([model.attrs['attr']]))
        ngrams = [unigrams]
        for n in range(2, model.attrs['ngram_size'] + 1):
            ngrams.append(model.ops.ngrams(n, unigrams))
        keys = model.ops.xp.concatenate(ngrams)
        (keys, vals) = model.ops.xp.unique(keys, return_counts=True)
        batch_keys.append(keys)
        batch_vals.append(vals)
    lengths = model.ops.asarray([arr.shape[0] for arr in batch_keys], dtype='int32')
    batch_keys = model.ops.xp.concatenate(batch_keys)
    batch_vals = model.ops.asarray(model.ops.xp.concatenate(batch_vals), dtype='f')

    def backprop(dY):
        if False:
            while True:
                i = 10
        return []
    return ((batch_keys, batch_vals, lengths), backprop)