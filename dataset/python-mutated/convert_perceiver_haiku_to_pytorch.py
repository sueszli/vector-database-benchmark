"""Convert Perceiver checkpoints originally implemented in Haiku."""
import argparse
import json
import pickle
from pathlib import Path
import haiku as hk
import numpy as np
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import PerceiverConfig, PerceiverForImageClassificationConvProcessing, PerceiverForImageClassificationFourier, PerceiverForImageClassificationLearned, PerceiverForMaskedLM, PerceiverForMultimodalAutoencoding, PerceiverForOpticalFlow, PerceiverImageProcessor, PerceiverTokenizer
from transformers.utils import logging
logging.set_verbosity_info()
logger = logging.get_logger(__name__)

def prepare_img():
    if False:
        i = 10
        return i + 15
    url = 'https://storage.googleapis.com/perceiver_io/dalmation.jpg'
    im = Image.open(requests.get(url, stream=True).raw)
    return im

def rename_keys(state_dict, architecture):
    if False:
        i = 10
        return i + 15
    for name in list(state_dict):
        param = state_dict.pop(name)
        name = name.replace('embed/embeddings', 'input_preprocessor.embeddings.weight')
        if name.startswith('trainable_position_encoding/pos_embs'):
            name = name.replace('trainable_position_encoding/pos_embs', 'input_preprocessor.position_embeddings.weight')
        name = name.replace('image_preprocessor/~/conv2_d/w', 'input_preprocessor.convnet_1x1.weight')
        name = name.replace('image_preprocessor/~/conv2_d/b', 'input_preprocessor.convnet_1x1.bias')
        name = name.replace('image_preprocessor/~_build_network_inputs/trainable_position_encoding/pos_embs', 'input_preprocessor.position_embeddings.position_embeddings')
        name = name.replace('image_preprocessor/~_build_network_inputs/position_encoding_projector/linear/w', 'input_preprocessor.positions_projection.weight')
        name = name.replace('image_preprocessor/~_build_network_inputs/position_encoding_projector/linear/b', 'input_preprocessor.positions_projection.bias')
        if 'counter' in name or 'hidden' in name:
            continue
        name = name.replace('image_preprocessor/~/conv2_d_downsample/~/conv/w', 'input_preprocessor.convnet.conv.weight')
        name = name.replace('image_preprocessor/~/conv2_d_downsample/~/batchnorm/offset', 'input_preprocessor.convnet.batchnorm.bias')
        name = name.replace('image_preprocessor/~/conv2_d_downsample/~/batchnorm/scale', 'input_preprocessor.convnet.batchnorm.weight')
        name = name.replace('image_preprocessor/~/conv2_d_downsample/~/batchnorm/~/mean_ema/average', 'input_preprocessor.convnet.batchnorm.running_mean')
        name = name.replace('image_preprocessor/~/conv2_d_downsample/~/batchnorm/~/var_ema/average', 'input_preprocessor.convnet.batchnorm.running_var')
        name = name.replace('image_preprocessor/patches_linear/b', 'input_preprocessor.conv_after_patches.bias')
        name = name.replace('image_preprocessor/patches_linear/w', 'input_preprocessor.conv_after_patches.weight')
        name = name.replace('multimodal_preprocessor/audio_mask_token/pos_embs', 'input_preprocessor.mask.audio')
        name = name.replace('multimodal_preprocessor/audio_padding/pos_embs', 'input_preprocessor.padding.audio')
        name = name.replace('multimodal_preprocessor/image_mask_token/pos_embs', 'input_preprocessor.mask.image')
        name = name.replace('multimodal_preprocessor/image_padding/pos_embs', 'input_preprocessor.padding.image')
        name = name.replace('multimodal_preprocessor/label_mask_token/pos_embs', 'input_preprocessor.mask.label')
        name = name.replace('multimodal_preprocessor/label_padding/pos_embs', 'input_preprocessor.padding.label')
        name = name.replace('multimodal_decoder/~/basic_decoder/cross_attention/', 'decoder.decoder.decoding_cross_attention.')
        name = name.replace('multimodal_decoder/~decoder_query/audio_padding/pos_embs', 'decoder.padding.audio')
        name = name.replace('multimodal_decoder/~decoder_query/image_padding/pos_embs', 'decoder.padding.image')
        name = name.replace('multimodal_decoder/~decoder_query/label_padding/pos_embs', 'decoder.padding.label')
        name = name.replace('multimodal_decoder/~/basic_decoder/output/b', 'decoder.decoder.final_layer.bias')
        name = name.replace('multimodal_decoder/~/basic_decoder/output/w', 'decoder.decoder.final_layer.weight')
        if architecture == 'multimodal_autoencoding':
            name = name.replace('classification_decoder/~/basic_decoder/~/trainable_position_encoding/pos_embs', 'decoder.modalities.label.decoder.output_position_encodings.position_embeddings')
        name = name.replace('flow_decoder/~/basic_decoder/cross_attention/', 'decoder.decoder.decoding_cross_attention.')
        name = name.replace('flow_decoder/~/basic_decoder/output/w', 'decoder.decoder.final_layer.weight')
        name = name.replace('flow_decoder/~/basic_decoder/output/b', 'decoder.decoder.final_layer.bias')
        name = name.replace('classification_decoder/~/basic_decoder/~/trainable_position_encoding/pos_embs', 'decoder.decoder.output_position_encodings.position_embeddings')
        name = name.replace('basic_decoder/~/trainable_position_encoding/pos_embs', 'decoder.output_position_encodings.position_embeddings')
        name = name.replace('classification_decoder/~/basic_decoder/cross_attention/', 'decoder.decoder.decoding_cross_attention.')
        name = name.replace('classification_decoder/~/basic_decoder/output/b', 'decoder.decoder.final_layer.bias')
        name = name.replace('classification_decoder/~/basic_decoder/output/w', 'decoder.decoder.final_layer.weight')
        name = name = name.replace('classification_decoder/~/basic_decoder/~/', 'decoder.decoder.')
        name = name.replace('basic_decoder/cross_attention/', 'decoder.decoding_cross_attention.')
        name = name.replace('basic_decoder/~/', 'decoder.')
        name = name.replace('projection_postprocessor/linear/b', 'output_postprocessor.modalities.image.classifier.bias')
        name = name.replace('projection_postprocessor/linear/w', 'output_postprocessor.modalities.image.classifier.weight')
        name = name.replace('classification_postprocessor/linear/b', 'output_postprocessor.modalities.label.classifier.bias')
        name = name.replace('classification_postprocessor/linear/w', 'output_postprocessor.modalities.label.classifier.weight')
        name = name.replace('audio_postprocessor/linear/b', 'output_postprocessor.modalities.audio.classifier.bias')
        name = name.replace('audio_postprocessor/linear/w', 'output_postprocessor.modalities.audio.classifier.weight')
        name = name.replace('perceiver_encoder/~/trainable_position_encoding/pos_embs', 'embeddings.latents')
        name = name.replace('encoder/~/trainable_position_encoding/pos_embs', 'embeddings.latents')
        if name.startswith('perceiver_encoder/~/'):
            if 'self_attention' in name:
                suffix = 'self_attends.'
            else:
                suffix = ''
            name = name.replace('perceiver_encoder/~/', 'encoder.' + suffix)
        if name.startswith('encoder/~/'):
            if 'self_attention' in name:
                suffix = 'self_attends.'
            else:
                suffix = ''
            name = name.replace('encoder/~/', 'encoder.' + suffix)
        if 'offset' in name:
            name = name.replace('offset', 'bias')
        if 'scale' in name:
            name = name.replace('scale', 'weight')
        if 'cross_attention' in name and 'layer_norm_2' in name:
            name = name.replace('layer_norm_2', 'layernorm')
        if 'self_attention' in name and 'layer_norm_1' in name:
            name = name.replace('layer_norm_1', 'layernorm')
        if 'cross_attention' in name and 'layer_norm_1' in name:
            name = name.replace('layer_norm_1', 'attention.self.layernorm2')
        if 'cross_attention' in name and 'layer_norm' in name:
            name = name.replace('layer_norm', 'attention.self.layernorm1')
        if 'self_attention' in name and 'layer_norm' in name:
            name = name.replace('layer_norm', 'attention.self.layernorm1')
        name = name.replace('-', '.')
        name = name.replace('/', '.')
        if ('cross_attention' in name or 'self_attention' in name) and 'mlp' not in name:
            if 'linear.b' in name:
                name = name.replace('linear.b', 'self.query.bias')
            if 'linear.w' in name:
                name = name.replace('linear.w', 'self.query.weight')
            if 'linear_1.b' in name:
                name = name.replace('linear_1.b', 'self.key.bias')
            if 'linear_1.w' in name:
                name = name.replace('linear_1.w', 'self.key.weight')
            if 'linear_2.b' in name:
                name = name.replace('linear_2.b', 'self.value.bias')
            if 'linear_2.w' in name:
                name = name.replace('linear_2.w', 'self.value.weight')
            if 'linear_3.b' in name:
                name = name.replace('linear_3.b', 'output.dense.bias')
            if 'linear_3.w' in name:
                name = name.replace('linear_3.w', 'output.dense.weight')
        if 'self_attention_' in name:
            name = name.replace('self_attention_', '')
        if 'self_attention' in name:
            name = name.replace('self_attention', '0')
        if 'mlp' in name:
            if 'linear.b' in name:
                name = name.replace('linear.b', 'dense1.bias')
            if 'linear.w' in name:
                name = name.replace('linear.w', 'dense1.weight')
            if 'linear_1.b' in name:
                name = name.replace('linear_1.b', 'dense2.bias')
            if 'linear_1.w' in name:
                name = name.replace('linear_1.w', 'dense2.weight')
        if name[-6:] == 'weight' and 'embeddings' not in name:
            param = np.transpose(param)
        if 'batchnorm' in name:
            param = np.squeeze(param)
        if 'embedding_decoder' not in name:
            state_dict['perceiver.' + name] = torch.from_numpy(param)
        else:
            state_dict[name] = torch.from_numpy(param)

@torch.no_grad()
def convert_perceiver_checkpoint(pickle_file, pytorch_dump_folder_path, architecture='MLM'):
    if False:
        while True:
            i = 10
    "\n    Copy/paste/tweak model's weights to our Perceiver structure.\n    "
    with open(pickle_file, 'rb') as f:
        checkpoint = pickle.loads(f.read())
    state = None
    if isinstance(checkpoint, dict) and architecture in ['image_classification', 'image_classification_fourier', 'image_classification_conv']:
        params = checkpoint['params']
        state = checkpoint['state']
    else:
        params = checkpoint
    state_dict = {}
    for (scope_name, parameters) in hk.data_structures.to_mutable_dict(params).items():
        for (param_name, param) in parameters.items():
            state_dict[scope_name + '/' + param_name] = param
    if state is not None:
        for (scope_name, parameters) in hk.data_structures.to_mutable_dict(state).items():
            for (param_name, param) in parameters.items():
                state_dict[scope_name + '/' + param_name] = param
    rename_keys(state_dict, architecture=architecture)
    config = PerceiverConfig()
    subsampling = None
    repo_id = 'huggingface/label-files'
    if architecture == 'MLM':
        config.qk_channels = 8 * 32
        config.v_channels = 1280
        model = PerceiverForMaskedLM(config)
    elif 'image_classification' in architecture:
        config.num_latents = 512
        config.d_latents = 1024
        config.d_model = 512
        config.num_blocks = 8
        config.num_self_attends_per_block = 6
        config.num_cross_attention_heads = 1
        config.num_self_attention_heads = 8
        config.qk_channels = None
        config.v_channels = None
        config.num_labels = 1000
        filename = 'imagenet-1k-id2label.json'
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type='dataset'), 'r'))
        id2label = {int(k): v for (k, v) in id2label.items()}
        config.id2label = id2label
        config.label2id = {v: k for (k, v) in id2label.items()}
        if architecture == 'image_classification':
            config.image_size = 224
            model = PerceiverForImageClassificationLearned(config)
        elif architecture == 'image_classification_fourier':
            config.d_model = 261
            model = PerceiverForImageClassificationFourier(config)
        elif architecture == 'image_classification_conv':
            config.d_model = 322
            model = PerceiverForImageClassificationConvProcessing(config)
        else:
            raise ValueError(f'Architecture {architecture} not supported')
    elif architecture == 'optical_flow':
        config.num_latents = 2048
        config.d_latents = 512
        config.d_model = 322
        config.num_blocks = 1
        config.num_self_attends_per_block = 24
        config.num_self_attention_heads = 16
        config.num_cross_attention_heads = 1
        model = PerceiverForOpticalFlow(config)
    elif architecture == 'multimodal_autoencoding':
        config.num_latents = 28 * 28 * 1
        config.d_latents = 512
        config.d_model = 704
        config.num_blocks = 1
        config.num_self_attends_per_block = 8
        config.num_self_attention_heads = 8
        config.num_cross_attention_heads = 1
        config.num_labels = 700
        images = torch.randn((1, 16, 3, 224, 224))
        audio = torch.randn((1, 30720, 1))
        nchunks = 128
        image_chunk_size = np.prod((16, 224, 224)) // nchunks
        audio_chunk_size = audio.shape[1] // config.samples_per_patch // nchunks
        chunk_idx = 0
        subsampling = {'image': torch.arange(image_chunk_size * chunk_idx, image_chunk_size * (chunk_idx + 1)), 'audio': torch.arange(audio_chunk_size * chunk_idx, audio_chunk_size * (chunk_idx + 1)), 'label': None}
        model = PerceiverForMultimodalAutoencoding(config)
        filename = 'kinetics700-id2label.json'
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type='dataset'), 'r'))
        id2label = {int(k): v for (k, v) in id2label.items()}
        config.id2label = id2label
        config.label2id = {v: k for (k, v) in id2label.items()}
    else:
        raise ValueError(f'Architecture {architecture} not supported')
    model.eval()
    model.load_state_dict(state_dict)
    input_mask = None
    if architecture == 'MLM':
        tokenizer = PerceiverTokenizer.from_pretrained('/Users/NielsRogge/Documents/Perceiver/Tokenizer files')
        text = 'This is an incomplete sentence where some words are missing.'
        encoding = tokenizer(text, padding='max_length', return_tensors='pt')
        encoding.input_ids[0, 51:60] = tokenizer.mask_token_id
        inputs = encoding.input_ids
        input_mask = encoding.attention_mask
    elif architecture in ['image_classification', 'image_classification_fourier', 'image_classification_conv']:
        image_processor = PerceiverImageProcessor()
        image = prepare_img()
        encoding = image_processor(image, return_tensors='pt')
        inputs = encoding.pixel_values
    elif architecture == 'optical_flow':
        inputs = torch.randn(1, 2, 27, 368, 496)
    elif architecture == 'multimodal_autoencoding':
        images = torch.randn((1, 16, 3, 224, 224))
        audio = torch.randn((1, 30720, 1))
        inputs = {'image': images, 'audio': audio, 'label': torch.zeros((images.shape[0], 700))}
    if architecture == 'multimodal_autoencoding':
        outputs = model(inputs=inputs, attention_mask=input_mask, subsampled_output_points=subsampling)
    else:
        outputs = model(inputs=inputs, attention_mask=input_mask)
    logits = outputs.logits
    if not isinstance(logits, dict):
        print('Shape of logits:', logits.shape)
    else:
        for (k, v) in logits.items():
            print(f'Shape of logits of modality {k}', v.shape)
    if architecture == 'MLM':
        expected_slice = torch.tensor([[-11.8336, -11.685, -11.8483], [-12.8149, -12.5863, -12.7904], [-12.844, -12.641, -12.8646]])
        assert torch.allclose(logits[0, :3, :3], expected_slice)
        masked_tokens_predictions = logits[0, 51:60].argmax(dim=-1).tolist()
        expected_list = [38, 115, 111, 121, 121, 111, 116, 109, 52]
        assert masked_tokens_predictions == expected_list
        print('Greedy predictions:')
        print(masked_tokens_predictions)
        print()
        print('Predicted string:')
        print(tokenizer.decode(masked_tokens_predictions))
    elif architecture in ['image_classification', 'image_classification_fourier', 'image_classification_conv']:
        print('Predicted class:', model.config.id2label[logits.argmax(-1).item()])
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f'Saving model to {pytorch_dump_folder_path}')
    model.save_pretrained(pytorch_dump_folder_path)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle_file', type=str, default=None, required=True, help="Path to local pickle file of a Perceiver checkpoint you'd like to convert.")
    parser.add_argument('--pytorch_dump_folder_path', default=None, type=str, required=True, help='Path to the output PyTorch model directory, provided as a string.')
    parser.add_argument('--architecture', default='MLM', type=str, help="\n        Architecture, provided as a string. One of 'MLM', 'image_classification', image_classification_fourier',\n        image_classification_fourier', 'optical_flow' or 'multimodal_autoencoding'.\n        ")
    args = parser.parse_args()
    convert_perceiver_checkpoint(args.pickle_file, args.pytorch_dump_folder_path, args.architecture)