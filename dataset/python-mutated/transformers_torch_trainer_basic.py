import evaluate
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from ray.train import ScalingConfig
from ray.train.huggingface.transformers import RayTrainReportCallback, prepare_trainer
from ray.train.torch import TorchTrainer

def train_func(config):
    if False:
        i = 10
        return i + 15
    dataset = load_dataset('yelp_review_full')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    def tokenize_function(examples):
        if False:
            print('Hello World!')
        return tokenizer(examples['text'], padding='max_length', truncation=True)
    tokenized_ds = dataset.map(tokenize_function, batched=True)
    small_train_ds = tokenized_ds['train'].shuffle(seed=42).select(range(1000))
    small_eval_ds = tokenized_ds['test'].shuffle(seed=42).select(range(1000))
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased', num_labels=5)
    metric = evaluate.load('accuracy')

    def compute_metrics(eval_pred):
        if False:
            for i in range(10):
                print('nop')
        (logits, labels) = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    training_args = TrainingArguments(output_dir='test_trainer', evaluation_strategy='epoch', report_to='none')
    trainer = Trainer(model=model, args=training_args, train_dataset=small_train_ds, eval_dataset=small_eval_ds, compute_metrics=compute_metrics)
    trainer.add_callback(RayTrainReportCallback())
    trainer = prepare_trainer(trainer)
    trainer.train()
trainer = TorchTrainer(train_func, scaling_config=ScalingConfig(num_workers=4, use_gpu=True))
trainer.fit()