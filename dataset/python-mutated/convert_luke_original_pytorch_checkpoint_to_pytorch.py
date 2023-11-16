"""Convert LUKE checkpoint."""
import argparse
import json
import os
import torch
from transformers import LukeConfig, LukeModel, LukeTokenizer, RobertaTokenizer
from transformers.tokenization_utils_base import AddedToken

@torch.no_grad()
def convert_luke_checkpoint(checkpoint_path, metadata_path, entity_vocab_path, pytorch_dump_folder_path, model_size):
    if False:
        print('Hello World!')
    with open(metadata_path) as metadata_file:
        metadata = json.load(metadata_file)
    config = LukeConfig(use_entity_aware_attention=True, **metadata['model_config'])
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    entity_vocab = load_entity_vocab(entity_vocab_path)
    tokenizer = RobertaTokenizer.from_pretrained(metadata['model_config']['bert_model_name'])
    entity_token_1 = AddedToken('<ent>', lstrip=False, rstrip=False)
    entity_token_2 = AddedToken('<ent2>', lstrip=False, rstrip=False)
    tokenizer.add_special_tokens({'additional_special_tokens': [entity_token_1, entity_token_2]})
    config.vocab_size += 2
    print(f'Saving tokenizer to {pytorch_dump_folder_path}')
    tokenizer.save_pretrained(pytorch_dump_folder_path)
    with open(os.path.join(pytorch_dump_folder_path, LukeTokenizer.vocab_files_names['entity_vocab_file']), 'w') as f:
        json.dump(entity_vocab, f)
    tokenizer = LukeTokenizer.from_pretrained(pytorch_dump_folder_path)
    word_emb = state_dict['embeddings.word_embeddings.weight']
    ent_emb = word_emb[tokenizer.convert_tokens_to_ids(['@'])[0]].unsqueeze(0)
    ent2_emb = word_emb[tokenizer.convert_tokens_to_ids(['#'])[0]].unsqueeze(0)
    state_dict['embeddings.word_embeddings.weight'] = torch.cat([word_emb, ent_emb, ent2_emb])
    for layer_index in range(config.num_hidden_layers):
        for matrix_name in ['query.weight', 'query.bias']:
            prefix = f'encoder.layer.{layer_index}.attention.self.'
            state_dict[prefix + 'w2e_' + matrix_name] = state_dict[prefix + matrix_name]
            state_dict[prefix + 'e2w_' + matrix_name] = state_dict[prefix + matrix_name]
            state_dict[prefix + 'e2e_' + matrix_name] = state_dict[prefix + matrix_name]
    entity_emb = state_dict['entity_embeddings.entity_embeddings.weight']
    entity_emb[entity_vocab['[MASK2]']] = entity_emb[entity_vocab['[MASK]']]
    model = LukeModel(config=config).eval()
    (missing_keys, unexpected_keys) = model.load_state_dict(state_dict, strict=False)
    if not (len(missing_keys) == 1 and missing_keys[0] == 'embeddings.position_ids'):
        raise ValueError(f"Missing keys {', '.join(missing_keys)}. Expected only missing embeddings.position_ids")
    if not all((key.startswith('entity_predictions') or key.startswith('lm_head') for key in unexpected_keys)):
        raise ValueError(f"Unexpected keys {', '.join([key for key in unexpected_keys if not (key.startswith('entity_predictions') or key.startswith('lm_head'))])}")
    tokenizer = LukeTokenizer.from_pretrained(pytorch_dump_folder_path, task='entity_classification')
    text = 'Top seed Ana Ivanovic said on Thursday she could hardly believe her luck as a fortuitous netcord helped the new world number one avoid a humiliating second- round exit at Wimbledon .'
    span = (39, 42)
    encoding = tokenizer(text, entity_spans=[span], add_prefix_space=True, return_tensors='pt')
    outputs = model(**encoding)
    if model_size == 'large':
        expected_shape = torch.Size((1, 42, 1024))
        expected_slice = torch.tensor([[0.0133, 0.0865, 0.0095], [0.3093, -0.2576, -0.7418], [-0.172, -0.2117, -0.2869]])
    else:
        expected_shape = torch.Size((1, 42, 768))
        expected_slice = torch.tensor([[0.0037, 0.1368, -0.0091], [0.1099, 0.3329, -0.1095], [0.0765, 0.5335, 0.1179]])
    if not outputs.last_hidden_state.shape == expected_shape:
        raise ValueError(f'Outputs.last_hidden_state.shape is {outputs.last_hidden_state.shape}, Expected shape is {expected_shape}')
    if not torch.allclose(outputs.last_hidden_state[0, :3, :3], expected_slice, atol=0.0001):
        raise ValueError
    if model_size == 'large':
        expected_shape = torch.Size((1, 1, 1024))
        expected_slice = torch.tensor([[0.0466, -0.0106, -0.0179]])
    else:
        expected_shape = torch.Size((1, 1, 768))
        expected_slice = torch.tensor([[0.1457, 0.1044, 0.0174]])
    if not outputs.entity_last_hidden_state.shape != expected_shape:
        raise ValueError(f'Outputs.entity_last_hidden_state.shape is {outputs.entity_last_hidden_state.shape}, Expected shape is {expected_shape}')
    if not torch.allclose(outputs.entity_last_hidden_state[0, :3, :3], expected_slice, atol=0.0001):
        raise ValueError
    print('Saving PyTorch model to {}'.format(pytorch_dump_folder_path))
    model.save_pretrained(pytorch_dump_folder_path)

def load_entity_vocab(entity_vocab_path):
    if False:
        while True:
            i = 10
    entity_vocab = {}
    with open(entity_vocab_path, 'r', encoding='utf-8') as f:
        for (index, line) in enumerate(f):
            (title, _) = line.rstrip().split('\t')
            entity_vocab[title] = index
    return entity_vocab
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, help='Path to a pytorch_model.bin file.')
    parser.add_argument('--metadata_path', default=None, type=str, help='Path to a metadata.json file, defining the configuration.')
    parser.add_argument('--entity_vocab_path', default=None, type=str, help='Path to an entity_vocab.tsv file, containing the entity vocabulary.')
    parser.add_argument('--pytorch_dump_folder_path', default=None, type=str, help='Path to where to dump the output PyTorch model.')
    parser.add_argument('--model_size', default='base', type=str, choices=['base', 'large'], help='Size of the model to be converted.')
    args = parser.parse_args()
    convert_luke_checkpoint(args.checkpoint_path, args.metadata_path, args.entity_vocab_path, args.pytorch_dump_folder_path, args.model_size)