import os
from dataclasses import dataclass, field
from trainer import Trainer, TrainerArgs
from TTS.config import load_config, register_config
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models import setup_model

@dataclass
class TrainTTSArgs(TrainerArgs):
    config_path: str = field(default=None, metadata={'help': 'Path to the config file.'})

def main():
    if False:
        return 10
    'Run `tts` model training directly by a `config.json` file.'
    train_args = TrainTTSArgs()
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
    (train_samples, eval_samples) = load_tts_samples(config.datasets, eval_split=True, eval_split_max_size=config.eval_split_max_size, eval_split_size=config.eval_split_size)
    model = setup_model(config, train_samples + eval_samples)
    trainer = Trainer(train_args, model.config, config.output_path, model=model, train_samples=train_samples, eval_samples=eval_samples, parse_command_line_args=False)
    trainer.fit()
if __name__ == '__main__':
    main()