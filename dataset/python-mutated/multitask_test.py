import os
import pytest
from allennlp.common import Params
from allennlp.common.testing import ModelTestCase
from allennlp.data import Instance, Vocabulary, Batch
from allennlp.data.fields import LabelField, TextField, MetadataField
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.models.heads import ClassifierHead
from allennlp.models import MultiTaskModel
from allennlp.modules.backbones import PretrainedTransformerBackbone
from allennlp.modules.seq2vec_encoders import ClsPooler

class TestMultiTaskModel(ModelTestCase):

    def test_forward_works(self):
        if False:
            while True:
                i = 10
        transformer_name = 'epwalsh/bert-xsmall-dummy'
        vocab = Vocabulary()
        backbone = PretrainedTransformerBackbone(vocab, transformer_name)
        head1 = ClassifierHead(vocab, seq2vec_encoder=ClsPooler(20), input_dim=20, num_labels=3)
        head2 = ClassifierHead(vocab, seq2vec_encoder=ClsPooler(20), input_dim=20, num_labels=4)
        model = MultiTaskModel(vocab, backbone, {'cls': head1})
        tokenizer = PretrainedTransformerTokenizer(model_name=transformer_name)
        token_indexers = PretrainedTransformerIndexer(model_name=transformer_name)
        tokens = tokenizer.tokenize('This is a test')
        text_field = TextField(tokens, {'tokens': token_indexers})
        label_field1 = LabelField(1, skip_indexing=True)
        label_field2 = LabelField(3, skip_indexing=True)
        instance = Instance({'text': text_field, 'label': label_field1, 'task': MetadataField('cls')})
        outputs = model.forward_on_instance(instance)
        assert 'encoded_text' in outputs
        assert 'cls_logits' in outputs
        assert 'loss' in outputs
        assert 'cls_loss' in outputs
        instance = Instance({'text': text_field, 'task': MetadataField('cls')})
        outputs = model.forward_on_instance(instance)
        assert 'encoded_text' in outputs
        assert 'cls_logits' in outputs
        assert 'loss' not in outputs
        model.eval()
        outputs = model.forward_on_instance(instance)
        assert 'encoded_text' in outputs
        assert 'loss' not in outputs
        assert 'cls_logits' in outputs
        model.train()
        model = MultiTaskModel(vocab, backbone, {'cls1': head1, 'cls2': head2}, arg_name_mapping={'backbone': {'question': 'text'}})
        instance1 = Instance({'text': text_field, 'label': label_field1, 'task': MetadataField('cls1')})
        instance2 = Instance({'text': text_field, 'label': label_field2, 'task': MetadataField('cls2')})
        batch = Batch([instance1, instance2])
        outputs = model.forward(**batch.as_tensor_dict())
        assert 'encoded_text' in outputs
        assert 'cls1_logits' in outputs
        assert 'cls1_loss' in outputs
        assert 'cls2_logits' in outputs
        assert 'cls2_loss' in outputs
        assert 'loss' in outputs
        combined_loss = outputs['cls1_loss'].item() + outputs['cls2_loss'].item()
        assert abs(outputs['loss'].item() - combined_loss) <= 1e-06
        instance = Instance({'text': text_field, 'label': label_field2, 'task': MetadataField('cls1')})
        with pytest.raises(IndexError):
            outputs = model.forward_on_instance(instance)
        instance = Instance({'question': text_field, 'text': text_field, 'task': MetadataField('cls1')})
        with pytest.raises(ValueError, match='duplicate argument text'):
            outputs = model.forward_on_instance(instance)

    def test_train_and_evaluate(self):
        if False:
            print('Hello World!')
        from allennlp.commands.train import train_model
        from allennlp.commands.evaluate import evaluate_from_args
        import argparse
        from allennlp.commands import Evaluate
        model_name = 'epwalsh/bert-xsmall-dummy'

        def reader():
            if False:
                for i in range(10):
                    print('nop')
            return {'type': 'text_classification_json', 'tokenizer': {'type': 'pretrained_transformer', 'model_name': model_name}, 'token_indexers': {'tokens': {'type': 'pretrained_transformer', 'model_name': model_name}}}

        def head():
            if False:
                for i in range(10):
                    print('nop')
            return {'type': 'classifier', 'seq2vec_encoder': {'type': 'cls_pooler', 'embedding_dim': 20}, 'input_dim': 20, 'num_labels': 2}
        head_eins_input = 'test_fixtures/data/text_classification_json/imdb_corpus.jsonl'
        head_zwei_input = 'test_fixtures/data/text_classification_json/ag_news_corpus_fake_sentiment_labels.jsonl'
        params = Params({'dataset_reader': {'type': 'multitask', 'readers': {'head_eins': reader(), 'head_zwei': reader()}}, 'model': {'type': 'multitask', 'backbone': {'type': 'pretrained_transformer', 'model_name': model_name}, 'heads': {'head_eins': head(), 'head_zwei': head()}, 'arg_name_mapping': {'backbone': {'tokens': 'text'}}}, 'train_data_path': {'head_eins': head_eins_input, 'head_zwei': head_zwei_input}, 'data_loader': {'type': 'multitask', 'scheduler': {'batch_size': 2}}, 'trainer': {'optimizer': {'type': 'huggingface_adamw', 'lr': 4e-05}, 'num_epochs': 2}})
        serialization_dir = os.path.join(self.TEST_DIR, 'serialization_dir')
        train_model(params, serialization_dir=serialization_dir)
        args = ['evaluate', str(self.TEST_DIR / 'serialization_dir'), f'{{"head_eins": "{head_eins_input}", "head_zwei": "{head_zwei_input}"}}', '--output-file', str(self.TEST_DIR / 'output.txt'), '--predictions-output-file', str(self.TEST_DIR / 'predictions.json')]
        parser = argparse.ArgumentParser(description='Testing')
        subparsers = parser.add_subparsers(title='Commands', metavar='')
        Evaluate().add_subparser(subparsers)
        args = parser.parse_args(args)
        metrics = evaluate_from_args(args)
        assert 'head_eins_accuracy' in metrics
        assert 'head_zwei_accuracy' in metrics