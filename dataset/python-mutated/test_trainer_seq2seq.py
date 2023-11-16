from transformers import AutoModelForSeq2SeqLM, BertTokenizer, DataCollatorForSeq2Seq, EncoderDecoderModel, GenerationConfig, Seq2SeqTrainer, Seq2SeqTrainingArguments, T5Tokenizer
from transformers.testing_utils import TestCasePlus, require_torch, slow
from transformers.utils import is_datasets_available
if is_datasets_available():
    import datasets

class Seq2seqTrainerTester(TestCasePlus):

    @slow
    @require_torch
    def test_finetune_bert2bert(self):
        if False:
            i = 10
            return i + 15
        bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained('prajjwal1/bert-tiny', 'prajjwal1/bert-tiny')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert2bert.config.vocab_size = bert2bert.config.encoder.vocab_size
        bert2bert.config.eos_token_id = tokenizer.sep_token_id
        bert2bert.config.decoder_start_token_id = tokenizer.cls_token_id
        bert2bert.config.max_length = 128
        train_dataset = datasets.load_dataset('cnn_dailymail', '3.0.0', split='train[:1%]')
        val_dataset = datasets.load_dataset('cnn_dailymail', '3.0.0', split='validation[:1%]')
        train_dataset = train_dataset.select(range(32))
        val_dataset = val_dataset.select(range(16))
        batch_size = 4

        def _map_to_encoder_decoder_inputs(batch):
            if False:
                for i in range(10):
                    print('nop')
            inputs = tokenizer(batch['article'], padding='max_length', truncation=True, max_length=512)
            outputs = tokenizer(batch['highlights'], padding='max_length', truncation=True, max_length=128)
            batch['input_ids'] = inputs.input_ids
            batch['attention_mask'] = inputs.attention_mask
            batch['decoder_input_ids'] = outputs.input_ids
            batch['labels'] = outputs.input_ids.copy()
            batch['labels'] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch['labels']]
            batch['decoder_attention_mask'] = outputs.attention_mask
            assert all((len(x) == 512 for x in inputs.input_ids))
            assert all((len(x) == 128 for x in outputs.input_ids))
            return batch

        def _compute_metrics(pred):
            if False:
                return 10
            labels_ids = pred.label_ids
            pred_ids = pred.predictions
            pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
            accuracy = sum([int(pred_str[i] == label_str[i]) for i in range(len(pred_str))]) / len(pred_str)
            return {'accuracy': accuracy}
        train_dataset = train_dataset.map(_map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size, remove_columns=['article', 'highlights'])
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'decoder_input_ids', 'decoder_attention_mask', 'labels'])
        val_dataset = val_dataset.map(_map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size, remove_columns=['article', 'highlights'])
        val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'decoder_input_ids', 'decoder_attention_mask', 'labels'])
        output_dir = self.get_auto_remove_tmp_dir()
        training_args = Seq2SeqTrainingArguments(output_dir=output_dir, per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size, predict_with_generate=True, evaluation_strategy='steps', do_train=True, do_eval=True, warmup_steps=0, eval_steps=2, logging_steps=2)
        trainer = Seq2SeqTrainer(model=bert2bert, args=training_args, compute_metrics=_compute_metrics, train_dataset=train_dataset, eval_dataset=val_dataset, tokenizer=tokenizer)
        trainer.train()

    @slow
    @require_torch
    def test_return_sequences(self):
        if False:
            i = 10
            return i + 15
        INPUT_COLUMN = 'question'
        TARGET_COLUMN = 'answer'
        MAX_INPUT_LENGTH = 256
        MAX_TARGET_LENGTH = 256
        dataset = datasets.load_dataset('gsm8k', 'main', split='train[:38]')
        model = AutoModelForSeq2SeqLM.from_pretrained('t5-small')
        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors='pt', padding='longest')
        gen_config = GenerationConfig.from_pretrained('t5-small', max_length=None, min_length=None, max_new_tokens=256, min_new_tokens=1, num_beams=5)
        training_args = Seq2SeqTrainingArguments('.', predict_with_generate=True)
        trainer = Seq2SeqTrainer(model=model, args=training_args, tokenizer=tokenizer, data_collator=data_collator, compute_metrics=lambda x: {'samples': x[0].shape[0]})

        def prepare_data(examples):
            if False:
                i = 10
                return i + 15
            inputs = examples[INPUT_COLUMN]
            targets = examples[TARGET_COLUMN]
            model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True)
            labels = tokenizer(text_target=targets, max_length=MAX_TARGET_LENGTH, truncation=True)
            model_inputs['labels'] = labels['input_ids']
            return model_inputs
        prepared_dataset = dataset.map(prepare_data, batched=True, remove_columns=[INPUT_COLUMN, TARGET_COLUMN])
        dataset_len = len(prepared_dataset)
        for num_return_sequences in range(3, 0, -1):
            gen_config.num_return_sequences = num_return_sequences
            metrics = trainer.evaluate(eval_dataset=prepared_dataset, generation_config=gen_config)
            assert metrics['eval_samples'] == dataset_len * num_return_sequences, f"Got {metrics['eval_samples']}, expected: {dataset_len * num_return_sequences}"