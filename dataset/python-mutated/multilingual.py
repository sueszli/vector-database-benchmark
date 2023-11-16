"""
Class for running multilingual pipelines
"""
from collections import OrderedDict
import copy
import logging
from typing import Union
from stanza.models.common.doc import Document
from stanza.models.common.utils import default_device
from stanza.pipeline.core import Pipeline, DownloadMethod
from stanza.pipeline._constants import *
from stanza.resources.common import DEFAULT_MODEL_DIR, get_language_resources, load_resources_json
logger = logging.getLogger('stanza')

class MultilingualPipeline:
    """
    Pipeline for handling multilingual data. Takes in text, detects language, and routes request to pipeline for that
    language.

    You can specify options to individual language pipelines with the lang_configs field.
    For example, if you want English pipelines to have NER, but want to turn that off for French, you can do:
        lang_configs = {"en": {"processors": "tokenize,pos,lemma,depparse,ner"},
                        "fr": {"processors": "tokenize,pos,lemma,depparse"}}
        pipeline = MultilingualPipeline(lang_configs=lang_configs)

    You can also pass in a defaultdict created in such a way that it provides default parameters for each language.
    For example, in order to only get tokenization for each language:
    (remembering that the Pipeline will automagically add MWT to a language which uses MWT):
        from collections import defaultdict
        lang_configs = defaultdict(lambda: dict(processors="tokenize"))
        pipeline = MultilingualPipeline(lang_configs=lang_configs)

    download_method can be set as in Pipeline to turn off downloading
      of the .json config or turn off downloading of everything
    """

    def __init__(self, model_dir: str=DEFAULT_MODEL_DIR, lang_id_config: dict=None, lang_configs: dict=None, ld_batch_size: int=64, max_cache_size: int=10, use_gpu: bool=None, restrict: bool=False, device: str=None, download_method: DownloadMethod=DownloadMethod.DOWNLOAD_RESOURCES, processors: Union[str, list]=None):
        if False:
            i = 10
            return i + 15
        self.model_dir = model_dir
        self.lang_id_config = {} if lang_id_config is None else copy.deepcopy(lang_id_config)
        self.lang_configs = {} if lang_configs is None else copy.deepcopy(lang_configs)
        self.max_cache_size = max_cache_size
        self.pipeline_cache = OrderedDict()
        if processors is None:
            self.default_processors = None
        elif isinstance(processors, str):
            self.default_processors = [x.strip() for x in processors.split(',')]
        else:
            self.default_processors = list(processors)
        self.download_method = download_method
        if 'download_method' not in self.lang_id_config:
            self.lang_id_config['download_method'] = self.download_method
        for lang in self.lang_configs:
            if 'lang' not in self.lang_configs[lang]:
                self.lang_configs[lang]['lang'] = lang
        if restrict and 'langid_lang_subset' not in self.lang_id_config:
            known_langs = sorted(self.lang_configs.keys())
            if known_langs == 0:
                logger.warning('MultilingualPipeline asked to restrict to lang_configs, but lang_configs was empty.  Ignoring...')
            else:
                logger.debug('Restricting MultilingualPipeline to %s', known_langs)
                self.lang_id_config['langid_lang_subset'] = known_langs
        if device is None:
            if use_gpu is None or use_gpu == True:
                device = default_device()
            else:
                device = 'cpu'
        self.device = device
        self.lang_id_pipeline = Pipeline(dir=self.model_dir, lang='multilingual', processors='langid', device=self.device, **self.lang_id_config)
        self.resources = load_resources_json(self.model_dir)

    def _update_pipeline_cache(self, lang):
        if False:
            return 10
        '\n        Do any necessary updates to the pipeline cache for this language. This includes building a new\n        pipeline for the lang, and possibly clearing out a language with the old last access date.\n        '
        if lang in self.pipeline_cache:
            self.pipeline_cache.move_to_end(lang, last=True)
        try:
            lang_config = self.lang_configs[lang]
        except KeyError:
            lang_config = {'lang': lang}
            self.lang_configs[lang] = lang_config
        if 'lang' not in lang_config:
            lang_config['lang'] = lang
        if 'download_method' not in lang_config:
            lang_config['download_method'] = self.download_method
        if 'processors' not in lang_config:
            if self.default_processors:
                lang_resources = get_language_resources(self.resources, lang)
                lang_processors = [x for x in self.default_processors if x in lang_resources]
                if lang_processors != self.default_processors:
                    logger.info('Not all requested processors %s available for %s.  Loading %s instead', self.default_processors, lang, lang_processors)
                lang_config['processors'] = ','.join(lang_processors)
        if lang not in self.pipeline_cache:
            logger.debug('Loading unknown language in MultilingualPipeline: %s', lang)
            if len(self.pipeline_cache) == self.max_cache_size:
                self.pipeline_cache.popitem(last=False)
            self.pipeline_cache[lang] = Pipeline(dir=self.model_dir, device=self.device, **self.lang_configs[lang])

    def process(self, doc):
        if False:
            for i in range(10):
                print('nop')
        '\n        Run language detection on a string, a Document, or a list of either, route to language specific pipeline\n        '
        singleton_input = not isinstance(doc, list)
        if singleton_input:
            docs = [doc]
        else:
            docs = doc
        if docs and isinstance(docs[0], str):
            docs = [Document([], text=text) for text in docs]
        docs_w_langid = self.lang_id_pipeline.process(docs)
        lang_batches = {}
        for (doc_idx, doc) in enumerate(docs_w_langid):
            logger.debug('Language for document %d: %s', doc_idx, doc.lang)
            if doc.lang not in lang_batches:
                lang_batches[doc.lang] = []
            lang_batches[doc.lang].append(doc)
        for lang in lang_batches.keys():
            self._update_pipeline_cache(lang)
            self.pipeline_cache[lang](lang_batches[lang])
        if singleton_input:
            return docs_w_langid[0]
        else:
            return docs_w_langid

    def __call__(self, doc):
        if False:
            i = 10
            return i + 15
        doc = self.process(doc)
        return doc