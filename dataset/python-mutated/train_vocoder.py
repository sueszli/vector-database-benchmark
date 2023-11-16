import os
from dataclasses import dataclass, field
from trainer import Trainer, TrainerArgs
from TTS.config import load_config, register_config
from TTS.utils.audio import AudioProcessor
from TTS.vocoder.datasets.preprocess import load_wav_data, load_wav_feat_data
from TTS.vocoder.models import setup_model

@dataclass
class TrainVocoderArgs(TrainerArgs):
    config_path: str = field(default=None, metadata={'help': 'Path to the config file.'})

def main():
    if False:
        i = 10
        return i + 15
    'Run `tts` model training directly by a `config.json` file.'
    train_args = TrainVocoderArgs()
    parser = train_args.init_argparse(arg_prefix='')
    (args, config_overrides) = parser.parse_known_args()
    train_args.parse_args(args)
    if args.config_path or args.continue_path:
        if args.config_path:
            config = load_config(args.config_path)
            if len(config_overrides) > 0:
                config.parse_known_args(config_overrides, relaxed_parser=True)
        elif args.continue_path:
            config = load_config(os.path.join(args.continue_path, 'config.json'))
            if len(config_overrides) > 0:
                config.parse_known_args(config_overrides, relaxed_parser=True)
        else:
            from TTS.config.shared_configs import BaseTrainingConfig
            config_base = BaseTrainingConfig()
            config_base.parse_known_args(config_overrides)
            config = register_config(config_base.model)()
    if 'feature_path' in config and config.feature_path:
        print(f' > Loading features from: {config.feature_path}')
        (eval_samples, train_samples) = load_wav_feat_data(config.data_path, config.feature_path, config.eval_split_size)
    else:
        (eval_samples, train_samples) = load_wav_data(config.data_path, config.eval_split_size)
    ap = AudioProcessor(**config.audio)
    model = setup_model(config)
    trainer = Trainer(train_args, config, config.output_path, model=model, train_samples=train_samples, eval_samples=eval_samples, training_assets={'audio_processor': ap}, parse_command_line_args=False)
    trainer.fit()
if __name__ == '__main__':
    main()