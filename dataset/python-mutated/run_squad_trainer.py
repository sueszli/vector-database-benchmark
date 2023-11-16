""" Fine-tuning the library models for question-answering."""
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import transformers
from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer, DataCollatorWithPadding, HfArgumentParser, SquadDataset, Trainer, TrainingArguments
from transformers import SquadDataTrainingArguments as DataTrainingArguments
from transformers.trainer_utils import is_main_process
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'})
    config_name: Optional[str] = field(default=None, metadata={'help': 'Pretrained config name or path if not the same as model_name'})
    tokenizer_name: Optional[str] = field(default=None, metadata={'help': 'Pretrained tokenizer name or path if not the same as model_name'})
    use_fast: bool = field(default=False, metadata={'help': 'Set this flag to use fast tokenization.'})
    cache_dir: Optional[str] = field(default=None, metadata={'help': 'Where do you want to store the pretrained models downloaded from huggingface.co'})

def main():
    if False:
        while True:
            i = 10
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        (model_args, data_args, training_args) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (model_args, data_args, training_args) = parser.parse_args_into_dataclasses()
    if os.path.exists(training_args.output_dir) and os.listdir(training_args.output_dir) and training_args.do_train and (not training_args.overwrite_output_dir):
        raise ValueError(f'Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.')
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN)
    logger.warning('Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s', training_args.local_rank, training_args.device, training_args.n_gpu, bool(training_args.local_rank != -1), training_args.fp16)
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info('Training/evaluation parameters %s', training_args)
    config = AutoConfig.from_pretrained(model_args.config_name if model_args.config_name else model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path, cache_dir=model_args.cache_dir, use_fast=False)
    model = AutoModelForQuestionAnswering.from_pretrained(model_args.model_name_or_path, from_tf=bool('.ckpt' in model_args.model_name_or_path), config=config, cache_dir=model_args.cache_dir)
    is_language_sensitive = hasattr(model.config, 'lang2id')
    train_dataset = SquadDataset(data_args, tokenizer=tokenizer, is_language_sensitive=is_language_sensitive, cache_dir=model_args.cache_dir) if training_args.do_train else None
    eval_dataset = SquadDataset(data_args, tokenizer=tokenizer, mode='dev', is_language_sensitive=is_language_sensitive, cache_dir=model_args.cache_dir) if training_args.do_eval else None
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8) if training_args.fp16 else None
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)
    if training_args.do_train:
        trainer.train(model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None)
        trainer.save_model()
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

def _mp_fn(index):
    if False:
        return 10
    main()
if __name__ == '__main__':
    main()