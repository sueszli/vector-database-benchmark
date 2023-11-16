"""
Hugging Face Transformers trainer wrapper module
"""
import os
import sys
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForQuestionAnswering, AutoModelForPreTraining, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq, Trainer, set_seed
from transformers import TrainingArguments as HFTrainingArguments
from ...data import Labels, Questions, Sequences, Texts
from ...models import Models, TokenDetection
from ..tensors import Tensors

class HFTrainer(Tensors):
    """
    Trains a new Hugging Face Transformer model using the Trainer framework.
    """

    def __call__(self, base, train, validation=None, columns=None, maxlength=None, stride=128, task='text-classification', prefix=None, metrics=None, tokenizers=None, checkpoint=None, **args):
        if False:
            print('Hello World!')
        '\n        Builds a new model using arguments.\n\n        Args:\n            base: path to base model, accepts Hugging Face model hub id, local path or (model, tokenizer) tuple\n            train: training data\n            validation: validation data\n            columns: tuple of columns to use for text/label, defaults to (text, None, label)\n            maxlength: maximum sequence length, defaults to tokenizer.model_max_length\n            stride: chunk size for splitting data for QA tasks\n            task: optional model task or category, determines the model type, defaults to "text-classification"\n            prefix: optional source prefix\n            metrics: optional function that computes and returns a dict of evaluation metrics\n            tokenizers: optional number of concurrent tokenizers, defaults to None\n            checkpoint: optional resume from checkpoint flag or path to checkpoint directory, defaults to None\n            args: training arguments\n\n        Returns:\n            (model, tokenizer)\n        '
        args = self.parse(args)
        set_seed(args.seed)
        (config, tokenizer, maxlength) = self.load(base, maxlength)
        (collator, labels) = (None, None)
        if task == 'language-generation':
            tokenizer.pad_token = tokenizer.pad_token if tokenizer.pad_token is not None else tokenizer.eos_token
            process = Texts(tokenizer, columns, maxlength)
            collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=8 if args.fp16 else None)
        elif task in ('language-modeling', 'token-detection'):
            process = Texts(tokenizer, columns, maxlength)
            collator = DataCollatorForLanguageModeling(tokenizer, pad_to_multiple_of=8 if args.fp16 else None)
        elif task == 'question-answering':
            process = Questions(tokenizer, columns, maxlength, stride)
        elif task == 'sequence-sequence':
            process = Sequences(tokenizer, columns, maxlength, prefix)
            collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8 if args.fp16 else None)
        else:
            process = Labels(tokenizer, columns, maxlength)
            labels = process.labels(train)
        (train, validation) = process(train, validation, os.cpu_count() if tokenizers and isinstance(tokenizers, bool) else tokenizers)
        model = self.model(task, base, config, labels, tokenizer)
        if collator:
            collator.model = model
        trainer = Trainer(model=model, tokenizer=tokenizer, data_collator=collator, args=args, train_dataset=train, eval_dataset=validation if validation else None, compute_metrics=metrics)
        trainer.train(resume_from_checkpoint=checkpoint)
        if validation:
            trainer.evaluate()
        if args.should_save:
            trainer.save_model()
            trainer.save_state()
        return (model.eval(), tokenizer)

    def parse(self, updates):
        if False:
            i = 10
            return i + 15
        '\n        Parses and merges custom arguments with defaults.\n\n        Args:\n            updates: custom arguments\n\n        Returns:\n            TrainingArguments\n        '
        args = {'output_dir': '', 'save_strategy': 'no', 'report_to': 'none', 'log_level': 'warning'}
        args.update(updates)
        return TrainingArguments(**args)

    def load(self, base, maxlength):
        if False:
            return 10
        '\n        Loads the base config and tokenizer.\n\n        Args:\n            base: base model - supports a file path or (model, tokenizer) tuple\n            maxlength: maximum sequence length\n\n        Returns:\n            (config, tokenizer, maxlength)\n        '
        if isinstance(base, (list, tuple)):
            (model, tokenizer) = base
            config = model.config
        else:
            config = AutoConfig.from_pretrained(base)
            tokenizer = AutoTokenizer.from_pretrained(base)
        Models.checklength(config, tokenizer)
        maxlength = min(maxlength if maxlength else sys.maxsize, tokenizer.model_max_length)
        return (config, tokenizer, maxlength)

    def model(self, task, base, config, labels, tokenizer):
        if False:
            print('Hello World!')
        '\n        Loads the base model to train.\n\n        Args:\n            task: optional model task or category, determines the model type, defaults to "text-classification"\n            base: base model - supports a file path or (model, tokenizer) tuple\n            config: model configuration\n            labels: number of labels\n            tokenizer: model tokenizer\n\n        Returns:\n            model\n        '
        if labels is not None:
            config.update({'num_labels': labels})
        if isinstance(base, (list, tuple)) and (not isinstance(base[0], str)):
            return base[0]
        if task == 'language-generation':
            return AutoModelForCausalLM.from_pretrained(base, config=config)
        if task == 'language-modeling':
            return AutoModelForMaskedLM.from_pretrained(base, config=config)
        if task == 'question-answering':
            return AutoModelForQuestionAnswering.from_pretrained(base, config=config)
        if task == 'sequence-sequence':
            return AutoModelForSeq2SeqLM.from_pretrained(base, config=config)
        if task == 'token-detection':
            return TokenDetection(AutoModelForMaskedLM.from_pretrained(base, config=config), AutoModelForPreTraining.from_pretrained(base, config=config), tokenizer)
        return AutoModelForSequenceClassification.from_pretrained(base, config=config)

class TrainingArguments(HFTrainingArguments):
    """
    Extends standard TrainingArguments to make the output directory optional for transient models.
    """

    @property
    def should_save(self):
        if False:
            while True:
                i = 10
        '\n        Override should_save to disable model saving when output directory is None.\n\n        Returns:\n            If model should be saved\n        '
        return super().should_save if self.output_dir else False