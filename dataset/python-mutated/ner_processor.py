"""
Processor for performing named entity tagging.
"""
import torch
import logging
from stanza.models.common import doc
from stanza.models.common.utils import unsort
from stanza.models.ner.data import DataLoader
from stanza.models.ner.trainer import Trainer
from stanza.models.ner.utils import merge_tags
from stanza.pipeline._constants import *
from stanza.pipeline.processor import UDProcessor, register_processor
logger = logging.getLogger('stanza')

@register_processor(name=NER)
class NERProcessor(UDProcessor):
    PROVIDES_DEFAULT = set([NER])
    REQUIRES_DEFAULT = set([TOKENIZE])

    def _get_dependencies(self, config, dep_name):
        if False:
            return 10
        dependencies = config.get(dep_name, None)
        if dependencies is not None:
            dependencies = dependencies.split(';')
            dependencies = [x if x else None for x in dependencies]
        else:
            dependencies = [x.get(dep_name) for x in config.get('dependencies', [])]
        return dependencies

    def _set_up_model(self, config, pipeline, device):
        if False:
            for i in range(10):
                print('nop')
        model_paths = config.get('model_path')
        if isinstance(model_paths, str):
            model_paths = model_paths.split(';')
        charlm_forward_files = self._get_dependencies(config, 'forward_charlm_path')
        charlm_backward_files = self._get_dependencies(config, 'backward_charlm_path')
        pretrain_files = self._get_dependencies(config, 'pretrain_path')
        self._predict_tagset = {}
        predict_tagset = config.get('predict_tagset', None)
        if predict_tagset:
            if isinstance(predict_tagset, int):
                self._predict_tagset[0] = predict_tagset
            else:
                predict_tagset = predict_tagset.split(';')
                for (piece_idx, piece) in enumerate(predict_tagset):
                    if piece:
                        self._predict_tagset[piece_idx] = int(piece)
        self.trainers = []
        for (model_path, pretrain_path, charlm_forward, charlm_backward) in zip(model_paths, pretrain_files, charlm_forward_files, charlm_backward_files):
            logger.debug('Loading %s with pretrain %s, forward charlm %s, backward charlm %s', model_path, pretrain_path, charlm_forward, charlm_backward)
            pretrain = pipeline.foundation_cache.load_pretrain(pretrain_path) if pretrain_path else None
            args = {'charlm_forward_file': charlm_forward, 'charlm_backward_file': charlm_backward}
            predict_tagset = self._predict_tagset.get(len(self.trainers), None)
            if predict_tagset is not None:
                args['predict_tagset'] = predict_tagset
            trainer = Trainer(args=args, model_file=model_path, pretrain=pretrain, device=device, foundation_cache=pipeline.foundation_cache)
            self.trainers.append(trainer)
        self._trainer = self.trainers[0]
        self.model_paths = model_paths

    def _set_up_final_config(self, config):
        if False:
            return 10
        ' Finalize the configurations for this processor, based off of values from a UD model. '
        if len(self.trainers) == 0:
            raise RuntimeError('Somehow there are no models loaded!')
        self._vocab = self.trainers[0].vocab
        self.configs = []
        for trainer in self.trainers:
            loaded_args = trainer.args
            loaded_args = {k: v for (k, v) in loaded_args.items() if not UDProcessor.filter_out_option(k)}
            loaded_args.update(config)
            self.configs.append(loaded_args)
        self._config = self.configs[0]

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return 'NERProcessor(%s)' % ';'.join(self.model_paths)

    def mark_inactive(self):
        if False:
            i = 10
            return i + 15
        ' Drop memory intensive resources if keeping this processor around for reasons other than running it. '
        super().mark_inactive()
        self.trainers = None

    def process(self, document):
        if False:
            i = 10
            return i + 15
        with torch.no_grad():
            all_preds = []
            for (trainer, config) in zip(self.trainers, self.configs):
                batch = DataLoader(document, config['batch_size'], config, vocab=trainer.vocab, evaluation=True, preprocess_tags=False, bert_tokenizer=trainer.model.bert_tokenizer)
                preds = []
                for (i, b) in enumerate(batch):
                    preds += trainer.predict(b)
                all_preds.append(preds)
        preds = [merge_tags(*x) for x in zip(*all_preds)]
        batch.doc.set([doc.NER], [y for x in preds for y in x], to_token=True)
        batch.doc.set([doc.MULTI_NER], [tuple(y) for x in zip(*all_preds) for y in zip(*x)], to_token=True)
        total = len(batch.doc.build_ents())
        logger.debug(f'{total} entities found in document.')
        return batch.doc

    def bulk_process(self, docs):
        if False:
            for i in range(10):
                print('nop')
        '\n        NER processor has a collation step after running inference\n        '
        docs = super().bulk_process(docs)
        for doc in docs:
            doc.build_ents()
        return docs

    def get_known_tags(self, model_idx=0):
        if False:
            return 10
        '\n        Return the tags known by this model\n\n        Removes the S-, B-, etc, and does not include O\n        Specify model_idx if the processor  has more than one model\n        '
        return self.trainers[model_idx].get_known_tags()