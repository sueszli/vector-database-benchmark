"""Convert Hubert checkpoint."""
import argparse
import torch
from transformers import UniSpeechSatConfig, UniSpeechSatForAudioFrameClassification, UniSpeechSatForSequenceClassification, UniSpeechSatForXVector, Wav2Vec2FeatureExtractor, logging
logging.set_verbosity_info()
logger = logging.get_logger(__name__)

def convert_classification(base_model_name, hf_config, downstream_dict):
    if False:
        i = 10
        return i + 15
    model = UniSpeechSatForSequenceClassification.from_pretrained(base_model_name, config=hf_config)
    model.projector.weight.data = downstream_dict['projector.weight']
    model.projector.bias.data = downstream_dict['projector.bias']
    model.classifier.weight.data = downstream_dict['model.post_net.linear.weight']
    model.classifier.bias.data = downstream_dict['model.post_net.linear.bias']
    return model

def convert_diarization(base_model_name, hf_config, downstream_dict):
    if False:
        i = 10
        return i + 15
    model = UniSpeechSatForAudioFrameClassification.from_pretrained(base_model_name, config=hf_config)
    model.classifier.weight.data = downstream_dict['model.linear.weight']
    model.classifier.bias.data = downstream_dict['model.linear.bias']
    return model

def convert_xvector(base_model_name, hf_config, downstream_dict):
    if False:
        i = 10
        return i + 15
    model = UniSpeechSatForXVector.from_pretrained(base_model_name, config=hf_config)
    model.projector.weight.data = downstream_dict['connector.weight']
    model.projector.bias.data = downstream_dict['connector.bias']
    for (i, kernel_size) in enumerate(hf_config.tdnn_kernel):
        model.tdnn[i].kernel.weight.data = downstream_dict[f'model.framelevel_feature_extractor.module.{i}.kernel.weight']
        model.tdnn[i].kernel.bias.data = downstream_dict[f'model.framelevel_feature_extractor.module.{i}.kernel.bias']
    model.feature_extractor.weight.data = downstream_dict['model.utterancelevel_feature_extractor.linear1.weight']
    model.feature_extractor.bias.data = downstream_dict['model.utterancelevel_feature_extractor.linear1.bias']
    model.classifier.weight.data = downstream_dict['model.utterancelevel_feature_extractor.linear2.weight']
    model.classifier.bias.data = downstream_dict['model.utterancelevel_feature_extractor.linear2.bias']
    model.objective.weight.data = downstream_dict['objective.W']
    return model

@torch.no_grad()
def convert_s3prl_checkpoint(base_model_name, config_path, checkpoint_path, model_dump_path):
    if False:
        return 10
    "\n    Copy/paste/tweak model's weights to transformers design.\n    "
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    downstream_dict = checkpoint['Downstream']
    hf_config = UniSpeechSatConfig.from_pretrained(config_path)
    hf_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(base_model_name, return_attention_mask=True, do_normalize=False)
    arch = hf_config.architectures[0]
    if arch.endswith('ForSequenceClassification'):
        hf_model = convert_classification(base_model_name, hf_config, downstream_dict)
    elif arch.endswith('ForAudioFrameClassification'):
        hf_model = convert_diarization(base_model_name, hf_config, downstream_dict)
    elif arch.endswith('ForXVector'):
        hf_model = convert_xvector(base_model_name, hf_config, downstream_dict)
    else:
        raise NotImplementedError(f'S3PRL weights conversion is not supported for {arch}')
    if hf_config.use_weighted_layer_sum:
        hf_model.layer_weights.data = checkpoint['Featurizer']['weights']
    hf_feature_extractor.save_pretrained(model_dump_path)
    hf_model.save_pretrained(model_dump_path)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_name', default=None, type=str, help='Name of the huggingface pretrained base model.')
    parser.add_argument('--config_path', default=None, type=str, help='Path to the huggingface classifier config.')
    parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to the s3prl checkpoint.')
    parser.add_argument('--model_dump_path', default=None, type=str, help='Path to the final converted model.')
    args = parser.parse_args()
    convert_s3prl_checkpoint(args.base_model_name, args.config_path, args.checkpoint_path, args.model_dump_path)