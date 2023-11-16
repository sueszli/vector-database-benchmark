""" Finetuning the library models for multiple choice (Bert, Roberta, XLNet)."""
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np
from utils_multiple_choice import MultipleChoiceDataset, Split, processors
import transformers
from transformers import AutoConfig, AutoModelForMultipleChoice, AutoTokenizer, DataCollatorWithPadding, EvalPrediction, HfArgumentParser, Trainer, TrainingArguments, set_seed
from transformers.trainer_utils import is_main_process
logger = logging.getLogger(__name__)

def simple_accuracy(preds, labels):
    if False:
        i = 10
        return i + 15
    return (preds == labels).mean()

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'})
    config_name: Optional[str] = field(default=None, metadata={'help': 'Pretrained config name or path if not the same as model_name'})
    tokenizer_name: Optional[str] = field(default=None, metadata={'help': 'Pretrained tokenizer name or path if not the same as model_name'})
    cache_dir: Optional[str] = field(default=None, metadata={'help': 'Where do you want to store the pretrained models downloaded from huggingface.co'})

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    task_name: str = field(metadata={'help': 'The name of the task to train on: ' + ', '.join(processors.keys())})
    data_dir: str = field(metadata={'help': 'Should contain the data files for the task.'})
    max_seq_length: int = field(default=128, metadata={'help': 'The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.'})
    overwrite_cache: bool = field(default=False, metadata={'help': 'Overwrite the cached training and evaluation sets'})

def main():
    if False:
        i = 10
        return i + 15
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
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
    set_seed(training_args.seed)
    try:
        processor = processors[data_args.task_name]()
        label_list = processor.get_labels()
        num_labels = len(label_list)
    except KeyError:
        raise ValueError('Task not found: %s' % data_args.task_name)
    config = AutoConfig.from_pretrained(model_args.config_name if model_args.config_name else model_args.model_name_or_path, num_labels=num_labels, finetuning_task=data_args.task_name, cache_dir=model_args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    model = AutoModelForMultipleChoice.from_pretrained(model_args.model_name_or_path, from_tf=bool('.ckpt' in model_args.model_name_or_path), config=config, cache_dir=model_args.cache_dir)
    train_dataset = MultipleChoiceDataset(data_dir=data_args.data_dir, tokenizer=tokenizer, task=data_args.task_name, max_seq_length=data_args.max_seq_length, overwrite_cache=data_args.overwrite_cache, mode=Split.train) if training_args.do_train else None
    eval_dataset = MultipleChoiceDataset(data_dir=data_args.data_dir, tokenizer=tokenizer, task=data_args.task_name, max_seq_length=data_args.max_seq_length, overwrite_cache=data_args.overwrite_cache, mode=Split.dev) if training_args.do_eval else None

    def compute_metrics(p: EvalPrediction) -> Dict:
        if False:
            return 10
        preds = np.argmax(p.predictions, axis=1)
        return {'acc': simple_accuracy(preds, p.label_ids)}
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8) if training_args.fp16 else None
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset, compute_metrics=compute_metrics, data_collator=data_collator)
    if training_args.do_train:
        trainer.train(model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None)
        trainer.save_model()
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)
    results = {}
    if training_args.do_eval:
        logger.info('*** Evaluate ***')
        result = trainer.evaluate()
        output_eval_file = os.path.join(training_args.output_dir, 'eval_results.txt')
        if trainer.is_world_master():
            with open(output_eval_file, 'w') as writer:
                logger.info('***** Eval results *****')
                for (key, value) in result.items():
                    logger.info('  %s = %s', key, value)
                    writer.write('%s = %s\n' % (key, value))
                results.update(result)
    return results

def _mp_fn(index):
    if False:
        for i in range(10):
            print('nop')
    main()
if __name__ == '__main__':
    main()