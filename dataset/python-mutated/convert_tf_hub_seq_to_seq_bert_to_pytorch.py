"""Convert Seq2Seq TF Hub checkpoint."""
import argparse
from . import BertConfig, BertGenerationConfig, BertGenerationDecoder, BertGenerationEncoder, load_tf_weights_in_bert_generation, logging
logging.set_verbosity_info()

def convert_tf_checkpoint_to_pytorch(tf_hub_path, pytorch_dump_path, is_encoder_named_decoder, vocab_size, is_encoder):
    if False:
        return 10
    bert_config = BertConfig.from_pretrained('bert-large-cased', vocab_size=vocab_size, max_position_embeddings=512, is_decoder=True, add_cross_attention=True)
    bert_config_dict = bert_config.to_dict()
    del bert_config_dict['type_vocab_size']
    config = BertGenerationConfig(**bert_config_dict)
    if is_encoder:
        model = BertGenerationEncoder(config)
    else:
        model = BertGenerationDecoder(config)
    print(f'Building PyTorch model from configuration: {config}')
    load_tf_weights_in_bert_generation(model, tf_hub_path, model_class='bert', is_encoder_named_decoder=is_encoder_named_decoder, is_encoder=is_encoder)
    print(f'Save PyTorch model and config to {pytorch_dump_path}')
    model.save_pretrained(pytorch_dump_path)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf_hub_path', default=None, type=str, required=True, help='Path to the TensorFlow checkpoint path.')
    parser.add_argument('--pytorch_dump_path', default=None, type=str, required=True, help='Path to the output PyTorch model.')
    parser.add_argument('--is_encoder_named_decoder', action='store_true', help='If decoder has to be renamed to encoder in PyTorch model.')
    parser.add_argument('--is_encoder', action='store_true', help='If model is an encoder.')
    parser.add_argument('--vocab_size', default=50358, type=int, help='Vocab size of model')
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(args.tf_hub_path, args.pytorch_dump_path, args.is_encoder_named_decoder, args.vocab_size, is_encoder=args.is_encoder)