from collections import Counter
from typing import List
import torch

def align_bpe_to_words(roberta, bpe_tokens: torch.LongTensor, other_tokens: List[str]):
    if False:
        while True:
            i = 10
    '\n    Helper to align GPT-2 BPE to other tokenization formats (e.g., spaCy).\n\n    Args:\n        roberta (RobertaHubInterface): RoBERTa instance\n        bpe_tokens (torch.LongTensor): GPT-2 BPE tokens of shape `(T_bpe)`\n        other_tokens (List[str]): other tokens of shape `(T_words)`\n\n    Returns:\n        List[str]: mapping from *other_tokens* to corresponding *bpe_tokens*.\n    '
    assert bpe_tokens.dim() == 1
    assert bpe_tokens[0] == 0

    def clean(text):
        if False:
            for i in range(10):
                print('nop')
        return text.strip()
    bpe_tokens = [roberta.task.source_dictionary.string([x]) for x in bpe_tokens]
    bpe_tokens = [clean(roberta.bpe.decode(x) if x not in {'<s>', ''} else x) for x in bpe_tokens]
    other_tokens = [clean(str(o)) for o in other_tokens]
    bpe_tokens = bpe_tokens[1:]
    assert ''.join(bpe_tokens) == ''.join(other_tokens)
    alignment = []
    bpe_toks = filter(lambda item: item[1] != '', enumerate(bpe_tokens, start=1))
    (j, bpe_tok) = next(bpe_toks)
    for other_tok in other_tokens:
        bpe_indices = []
        while True:
            if other_tok.startswith(bpe_tok):
                bpe_indices.append(j)
                other_tok = other_tok[len(bpe_tok):]
                try:
                    (j, bpe_tok) = next(bpe_toks)
                except StopIteration:
                    (j, bpe_tok) = (None, None)
            elif bpe_tok.startswith(other_tok):
                bpe_indices.append(j)
                bpe_tok = bpe_tok[len(other_tok):]
                other_tok = ''
            else:
                raise Exception('Cannot align "{}" and "{}"'.format(other_tok, bpe_tok))
            if other_tok == '':
                break
        assert len(bpe_indices) > 0
        alignment.append(bpe_indices)
    assert len(alignment) == len(other_tokens)
    return alignment

def align_features_to_words(roberta, features, alignment):
    if False:
        i = 10
        return i + 15
    '\n    Align given features to words.\n\n    Args:\n        roberta (RobertaHubInterface): RoBERTa instance\n        features (torch.Tensor): features to align of shape `(T_bpe x C)`\n        alignment: alignment between BPE tokens and words returned by\n            func:`align_bpe_to_words`.\n    '
    assert features.dim() == 2
    bpe_counts = Counter((j for bpe_indices in alignment for j in bpe_indices))
    assert bpe_counts[0] == 0
    denom = features.new([bpe_counts.get(j, 1) for j in range(len(features))])
    weighted_features = features / denom.unsqueeze(-1)
    output = [weighted_features[0]]
    largest_j = -1
    for bpe_indices in alignment:
        output.append(weighted_features[bpe_indices].sum(dim=0))
        largest_j = max(largest_j, *bpe_indices)
    for j in range(largest_j + 1, len(features)):
        output.append(weighted_features[j])
    output = torch.stack(output)
    assert torch.all(torch.abs(output.sum(dim=0) - features.sum(dim=0)) < 0.0001)
    return output

def spacy_nlp():
    if False:
        return 10
    if getattr(spacy_nlp, '_nlp', None) is None:
        try:
            from spacy.lang.en import English
            spacy_nlp._nlp = English()
        except ImportError:
            raise ImportError('Please install spacy with: pip install spacy')
    return spacy_nlp._nlp

def spacy_tokenizer():
    if False:
        for i in range(10):
            print('nop')
    if getattr(spacy_tokenizer, '_tokenizer', None) is None:
        try:
            nlp = spacy_nlp()
            spacy_tokenizer._tokenizer = nlp.Defaults.create_tokenizer(nlp)
        except ImportError:
            raise ImportError('Please install spacy with: pip install spacy')
    return spacy_tokenizer._tokenizer