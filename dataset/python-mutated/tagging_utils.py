"""Utilities for sequence tagging tasks for entity-level tasks (e.g., NER)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

def get_span_labels(sentence_tags, inv_label_mapping=None):
    if False:
        for i in range(10):
            print('nop')
    'Go from token-level labels to list of entities (start, end, class).'
    if inv_label_mapping:
        sentence_tags = [inv_label_mapping[i] for i in sentence_tags]
    span_labels = []
    last = 'O'
    start = -1
    for (i, tag) in enumerate(sentence_tags):
        (pos, _) = (None, 'O') if tag == 'O' else tag.split('-')
        if (pos == 'S' or pos == 'B' or tag == 'O') and last != 'O':
            span_labels.append((start, i - 1, last.split('-')[-1]))
        if pos == 'B' or pos == 'S' or last == 'O':
            start = i
        last = tag
    if sentence_tags[-1] != 'O':
        span_labels.append((start, len(sentence_tags) - 1, sentence_tags[-1].split('-')[-1]))
    return span_labels

def get_tags(span_labels, length, encoding):
    if False:
        i = 10
        return i + 15
    'Converts a list of entities to token-label labels based on the provided\n  encoding (e.g., BIOES).\n  '
    tags = ['O' for _ in range(length)]
    for (s, e, t) in span_labels:
        for i in range(s, e + 1):
            tags[i] = 'I-' + t
        if 'E' in encoding:
            tags[e] = 'E-' + t
        if 'B' in encoding:
            tags[s] = 'B-' + t
        if 'S' in encoding and s == e:
            tags[s] = 'S-' + t
    return tags