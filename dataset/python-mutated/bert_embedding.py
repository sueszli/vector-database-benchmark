import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pack_sequence, PackedSequence
logger = logging.getLogger('stanza')
BERT_ARGS = {'vinai/phobert-base': {'use_fast': True}, 'vinai/phobert-large': {'use_fast': True}}

class TextTooLongError(ValueError):
    """
    A text was too long for the underlying model (possibly BERT)
    """

    def __init__(self, length, max_len, line_num, text):
        if False:
            return 10
        super().__init__('Found a text of length %d (possibly after tokenizing).  Maximum handled length is %d  Error occurred at line %d' % (length, max_len, line_num))
        self.line_num = line_num
        self.text = text

def update_max_length(model_name, tokenizer):
    if False:
        while True:
            i = 10
    if model_name in ('google/muril-base-cased', 'airesearch/wangchanberta-base-att-spm-uncased', 'camembert/camembert-large'):
        tokenizer.model_max_length = 512

def load_tokenizer(model_name):
    if False:
        while True:
            i = 10
    if model_name:
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError('Please install transformers library for BERT support! Try `pip install transformers`.')
        bert_args = BERT_ARGS.get(model_name, dict())
        if not model_name.startswith('vinai/phobert'):
            bert_args['add_prefix_space'] = True
        bert_tokenizer = AutoTokenizer.from_pretrained(model_name, **bert_args)
        update_max_length(model_name, bert_tokenizer)
        return bert_tokenizer
    return None

def load_bert(model_name):
    if False:
        for i in range(10):
            print('nop')
    if model_name:
        try:
            from transformers import AutoModel
        except ImportError:
            raise ImportError('Please install transformers library for BERT support! Try `pip install transformers`.')
        bert_model = AutoModel.from_pretrained(model_name)
        bert_tokenizer = load_tokenizer(model_name)
        return (bert_model, bert_tokenizer)
    return (None, None)

def tokenize_manual(model_name, sent, tokenizer):
    if False:
        i = 10
        return i + 15
    '\n    Tokenize a sentence manually, using for checking long sentences and PHOBert.\n    '
    tokenized = [word.replace('\xa0', '_').replace(' ', '_') for word in sent] if model_name.startswith('vinai/phobert') else [word.replace('\xa0', ' ') for word in sent]
    sentence = ' '.join(tokenized)
    tokenized = tokenizer.tokenize(sentence)
    sent_ids = tokenizer.convert_tokens_to_ids(tokenized)
    tokenized_sent = [tokenizer.bos_token_id] + sent_ids + [tokenizer.eos_token_id]
    return (tokenized, tokenized_sent)

def filter_data(model_name, data, tokenizer=None, log_level=logging.DEBUG):
    if False:
        while True:
            i = 10
    '\n    Filter out the (NER, POS) data that is too long for BERT model.\n    '
    if tokenizer is None:
        tokenizer = load_tokenizer(model_name)
    filtered_data = []
    for sent in data:
        sentence = [word if isinstance(word, str) else word[0] for word in sent]
        (_, tokenized_sent) = tokenize_manual(model_name, sentence, tokenizer)
        if len(tokenized_sent) > tokenizer.model_max_length - 2:
            continue
        filtered_data.append(sent)
    logger.log(log_level, 'Eliminated %d of %d datapoints because their length is over maximum size of BERT model.', len(data) - len(filtered_data), len(data))
    return filtered_data

def cloned_feature(feature, num_layers, detach=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Clone & detach the feature, keeping the last N layers (or averaging -2,-3,-4 if not specified)\n\n    averaging 3 of the last 4 layers worked well for non-VI languages\n    '
    if num_layers is None:
        feature = torch.stack(feature[-4:-1], axis=3).sum(axis=3) / 4
    else:
        feature = torch.stack(feature[-num_layers:], axis=3)
    if detach:
        return feature.clone().detach()
    else:
        return feature

def extract_bart_word_embeddings(model_name, tokenizer, model, data, device, keep_endpoints, num_layers, detach=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Handles vi-bart.  May need testing before using on other bart\n\n    https://github.com/VinAIResearch/BARTpho\n    '
    processed = []
    sentences = [' '.join([word.replace(' ', '_') for word in sentence]) for sentence in data]
    tokenized = tokenizer(sentences, return_tensors='pt', padding=True, return_attention_mask=True)
    input_ids = tokenized['input_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)
    for i in range(int(math.ceil(len(sentences) / 128))):
        start_sentence = i * 128
        end_sentence = min(start_sentence + 128, len(sentences))
        input_ids = input_ids[start_sentence:end_sentence]
        attention_mask = attention_mask[start_sentence:end_sentence]
        if detach:
            with torch.no_grad():
                features = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                features = cloned_feature(features.decoder_hidden_states, num_layers, detach)
        else:
            features = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            features = cloned_feature(features.decoder_hidden_states, num_layers, detach)
        for (feature, sentence) in zip(features, data):
            feature = feature[:len(sentence) + 2]
            if not keep_endpoints:
                feature = feature[1:-1]
            processed.append(feature)
    return processed

def extract_phobert_embeddings(model_name, tokenizer, model, data, device, keep_endpoints, num_layers, detach=True):
    if False:
        return 10
    "\n    Extract transformer embeddings using a method specifically for phobert\n\n    Since phobert doesn't have the is_split_into_words / tokenized.word_ids(batch_index=0)\n    capability, we instead look for @@ to denote a continued token.\n    data: list of list of string (the text tokens)\n    "
    processed = []
    tokenized_sents = []
    list_tokenized = []
    for (idx, sent) in enumerate(data):
        (tokenized, tokenized_sent) = tokenize_manual(model_name, sent, tokenizer)
        list_tokenized.append(tokenized)
        if len(tokenized_sent) > tokenizer.model_max_length:
            logger.error('Invalid size, max size: %d, got %d %s', tokenizer.model_max_length, len(tokenized_sent), data[idx])
            raise TextTooLongError(len(tokenized_sent), tokenizer.model_max_length, idx, ' '.join(data[idx]))
        tokenized_sents.append(torch.tensor(tokenized_sent).detach())
        processed_sent = []
        processed.append(processed_sent)
    size = len(tokenized_sents)
    tokenized_sents_padded = torch.nn.utils.rnn.pad_sequence(tokenized_sents, batch_first=True, padding_value=tokenizer.pad_token_id)
    features = []
    for i in range(int(math.ceil(size / 128))):
        padded_input = tokenized_sents_padded[128 * i:128 * i + 128]
        start_sentence = i * 128
        end_sentence = start_sentence + padded_input.shape[0]
        attention_mask = torch.zeros(end_sentence - start_sentence, padded_input.shape[1], device=device)
        for (sent_idx, sent) in enumerate(tokenized_sents[start_sentence:end_sentence]):
            attention_mask[sent_idx, :len(sent)] = 1
        if detach:
            with torch.no_grad():
                feature = model(padded_input.clone().detach().to(device), attention_mask=attention_mask, output_hidden_states=True)
                features += cloned_feature(feature.hidden_states, num_layers, detach)
        else:
            feature = model(padded_input.to(device), attention_mask=attention_mask, output_hidden_states=True)
            features += cloned_feature(feature.hidden_states, num_layers, detach)
    assert len(features) == size
    assert len(features) == len(processed)
    offsets = [[idx2 + 1 for (idx2, _) in enumerate(list_tokenized[idx]) if idx2 > 0 and (not list_tokenized[idx][idx2 - 1].endswith('@@')) or idx2 == 0] for (idx, sent) in enumerate(processed)]
    if keep_endpoints:
        offsets = [[0] + off + [-1] for off in offsets]
    processed = [feature[offset] for (feature, offset) in zip(features, offsets)]
    return processed
BAD_TOKENIZERS = ('bert-base-german-cased', 'dbmdz/bert-base-german-cased', 'dbmdz/bert-base-italian-xxl-cased', 'dbmdz/bert-base-italian-cased', 'dbmdz/electra-base-italian-xxl-cased-discriminator', 'avichr/heBERT', 'onlplab/alephbert-base', 'imvladikon/alephbertgimmel-base-512', 'cahya/bert-base-indonesian-1.5G', 'indolem/indobert-base-uncased', 'google/muril-base-cased', 'l3cube-pune/marathi-roberta')

def fix_blank_tokens(tokenizer, data):
    if False:
        return 10
    'Patch bert tokenizers with missing characters\n\n    There is an issue that some tokenizers (so far the German ones identified above)\n    tokenize soft hyphens or other unknown characters into nothing\n    If an entire word is tokenized as a soft hyphen, this means the tokenizer\n    simply vaporizes that word.  The result is we\'re missing an embedding for\n    an entire word we wanted to use.\n\n    The solution we take here is to look for any words which get vaporized\n    in such a manner, eg `len(token) == 2`, and replace it with a regular "-"\n\n    Actually, recently we have found that even the Bert / Electra tokenizer\n    can do this in the case of "words" which are one special character long,\n    so the easiest thing to do is just always run this function\n    '
    new_data = []
    for sentence in data:
        tokenized = tokenizer(sentence, is_split_into_words=False).input_ids
        new_sentence = [word if len(token) > 2 else '-' for (word, token) in zip(sentence, tokenized)]
        new_data.append(new_sentence)
    return new_data

def extract_xlnet_embeddings(model_name, tokenizer, model, data, device, keep_endpoints, num_layers, detach=True):
    if False:
        while True:
            i = 10
    tokenized = tokenizer(data, is_split_into_words=True, return_offsets_mapping=False, return_attention_mask=False)
    list_offsets = [[None] * (len(sentence) + 2) for sentence in data]
    for idx in range(len(data)):
        offsets = tokenized.word_ids(batch_index=idx)
        list_offsets[idx][0] = 0
        for (pos, offset) in enumerate(offsets):
            if offset is None:
                break
            list_offsets[idx][offset + 1] = pos + 1
        list_offsets[idx][-1] = list_offsets[idx][-2] + 1
        if any((x is None for x in list_offsets[idx])):
            raise ValueError('OOPS, hit None when preparing to use Bert\ndata[idx]: {}\noffsets: {}\nlist_offsets[idx]: {}'.format(data[idx], offsets, list_offsets[idx], tokenized))
        if len(offsets) > tokenizer.model_max_length - 2:
            logger.error('Invalid size, max size: %d, got %d %s', tokenizer.model_max_length, len(offsets), data[idx])
            raise TextTooLongError(len(offsets), tokenizer.model_max_length, idx, ' '.join(data[idx]))
    features = []
    for i in range(int(math.ceil(len(data) / 128))):
        input_ids = [[tokenizer.bos_token_id] + x[:-2] + [tokenizer.eos_token_id] for x in tokenized['input_ids'][128 * i:128 * i + 128]]
        max_len = max((len(x) for x in input_ids))
        attention_mask = torch.zeros(len(input_ids), max_len, dtype=torch.long, device=device)
        for (idx, input_row) in enumerate(input_ids):
            attention_mask[idx, :len(input_row)] = 1
            if len(input_row) < max_len:
                input_row.extend([tokenizer.pad_token_id] * (max_len - len(input_row)))
        if detach:
            with torch.no_grad():
                id_tensor = torch.tensor(input_ids, device=device)
                feature = model(id_tensor, attention_mask=attention_mask, output_hidden_states=True)
                features += cloned_feature(feature.hidden_states, num_layers, detach)
        else:
            id_tensor = torch.tensor(input_ids, device=device)
            feature = model(id_tensor, attention_mask=attention_mask, output_hidden_states=True)
            features += cloned_feature(feature.hidden_states, num_layers, detach)
    processed = []
    if not keep_endpoints:
        list_offsets = [sent[1:-1] for sent in list_offsets]
    for (feature, offsets) in zip(features, list_offsets):
        new_sent = feature[offsets]
        processed.append(new_sent)
    return processed

def extract_bert_embeddings(model_name, tokenizer, model, data, device, keep_endpoints, num_layers=None, detach=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Extract transformer embeddings using a generic roberta extraction\n\n    data: list of list of string (the text tokens)\n    num_layers: how many to return.  If None, the average of -2, -3, -4 is returned\n    '
    if model_name.startswith('vinai/phobert'):
        return extract_phobert_embeddings(model_name, tokenizer, model, data, device, keep_endpoints, num_layers, detach)
    if 'bart' in model_name:
        return extract_bart_word_embeddings(model_name, tokenizer, model, data, device, keep_endpoints, num_layers, detach)
    if isinstance(data, tuple):
        data = list(data)
    if 'xlnet' in model_name:
        return extract_xlnet_embeddings(model_name, tokenizer, model, data, device, keep_endpoints, num_layers, detach)
    data = fix_blank_tokens(tokenizer, data)
    tokenized = tokenizer(data, padding='longest', is_split_into_words=True, return_offsets_mapping=False, return_attention_mask=True)
    list_offsets = [[None] * (len(sentence) + 2) for sentence in data]
    for idx in range(len(data)):
        offsets = tokenized.word_ids(batch_index=idx)
        for (pos, offset) in enumerate(offsets):
            if offset is None:
                continue
            list_offsets[idx][offset + 1] = pos
        list_offsets[idx][0] = 0
        list_offsets[idx][-1] = list_offsets[idx][-2] + 1
        if any((x is None for x in list_offsets[idx])):
            raise ValueError('OOPS, hit None when preparing to use Bert\ndata[idx]: {}\noffsets: {}\nlist_offsets[idx]: {}'.format(data[idx], offsets, list_offsets[idx], tokenized))
        if len(offsets) > tokenizer.model_max_length - 2:
            logger.error('Invalid size, max size: %d, got %d.\nTokens: %s\nTokenized: %s', tokenizer.model_max_length, len(offsets), data[idx], offsets)
            raise TextTooLongError(len(offsets), tokenizer.model_max_length, idx, ' '.join(data[idx]))
    features = []
    for i in range(int(math.ceil(len(data) / 128))):
        if detach:
            with torch.no_grad():
                attention_mask = torch.tensor(tokenized['attention_mask'][128 * i:128 * i + 128], device=device)
                id_tensor = torch.tensor(tokenized['input_ids'][128 * i:128 * i + 128], device=device)
                feature = model(id_tensor, attention_mask=attention_mask, output_hidden_states=True)
                features += cloned_feature(feature.hidden_states, num_layers, detach)
        else:
            attention_mask = torch.tensor(tokenized['attention_mask'][128 * i:128 * i + 128], device=device)
            id_tensor = torch.tensor(tokenized['input_ids'][128 * i:128 * i + 128], device=device)
            feature = model(id_tensor, attention_mask=attention_mask, output_hidden_states=True)
            features += cloned_feature(feature.hidden_states, num_layers, detach)
    processed = []
    if not keep_endpoints:
        list_offsets = [sent[1:-1] for sent in list_offsets]
    for (feature, offsets) in zip(features, list_offsets):
        new_sent = feature[offsets]
        processed.append(new_sent)
    return processed