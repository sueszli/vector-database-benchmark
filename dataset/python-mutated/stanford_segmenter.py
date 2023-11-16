import json
import os
import tempfile
import warnings
from subprocess import PIPE
from nltk.internals import _java_options, config_java, find_dir, find_file, find_jar, java
from nltk.tokenize.api import TokenizerI
_stanford_url = 'https://nlp.stanford.edu/software'

class StanfordSegmenter(TokenizerI):
    """Interface to the Stanford Segmenter

    If stanford-segmenter version is older than 2016-10-31, then path_to_slf4j
    should be provieded, for example::

        seg = StanfordSegmenter(path_to_slf4j='/YOUR_PATH/slf4j-api.jar')

    >>> from nltk.tokenize.stanford_segmenter import StanfordSegmenter
    >>> seg = StanfordSegmenter() # doctest: +SKIP
    >>> seg.default_config('zh') # doctest: +SKIP
    >>> sent = u'这是斯坦福中文分词器测试'
    >>> print(seg.segment(sent)) # doctest: +SKIP
    这 是 斯坦福 中文 分词器 测试
    <BLANKLINE>
    >>> seg.default_config('ar') # doctest: +SKIP
    >>> sent = u'هذا هو تصنيف ستانفورد العربي للكلمات'
    >>> print(seg.segment(sent.split())) # doctest: +SKIP
    هذا هو تصنيف ستانفورد العربي ل الكلمات
    <BLANKLINE>
    """
    _JAR = 'stanford-segmenter.jar'

    def __init__(self, path_to_jar=None, path_to_slf4j=None, java_class=None, path_to_model=None, path_to_dict=None, path_to_sihan_corpora_dict=None, sihan_post_processing='false', keep_whitespaces='false', encoding='UTF-8', options=None, verbose=False, java_options='-mx2g'):
        if False:
            i = 10
            return i + 15
        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn(str("\nThe StanfordTokenizer will be deprecated in version 3.2.5.\nPlease use \x1b[91mnltk.parse.corenlp.CoreNLPTokenizer\x1b[0m instead.'"), DeprecationWarning, stacklevel=2)
        warnings.simplefilter('ignore', DeprecationWarning)
        stanford_segmenter = find_jar(self._JAR, path_to_jar, env_vars=('STANFORD_SEGMENTER',), searchpath=(), url=_stanford_url, verbose=verbose)
        if path_to_slf4j is not None:
            slf4j = find_jar('slf4j-api.jar', path_to_slf4j, env_vars=('SLF4J', 'STANFORD_SEGMENTER'), searchpath=(), url=_stanford_url, verbose=verbose)
        else:
            slf4j = None
        self._stanford_jar = os.pathsep.join((_ for _ in [stanford_segmenter, slf4j] if _ is not None))
        self._java_class = java_class
        self._model = path_to_model
        self._sihan_corpora_dict = path_to_sihan_corpora_dict
        self._sihan_post_processing = sihan_post_processing
        self._keep_whitespaces = keep_whitespaces
        self._dict = path_to_dict
        self._encoding = encoding
        self.java_options = java_options
        options = {} if options is None else options
        self._options_cmd = ','.join((f'{key}={json.dumps(val)}' for (key, val) in options.items()))

    def default_config(self, lang):
        if False:
            for i in range(10):
                print('nop')
        '\n        Attempt to initialize Stanford Word Segmenter for the specified language\n        using the STANFORD_SEGMENTER and STANFORD_MODELS environment variables\n        '
        search_path = ()
        if os.environ.get('STANFORD_SEGMENTER'):
            search_path = {os.path.join(os.environ.get('STANFORD_SEGMENTER'), 'data')}
        self._dict = None
        self._sihan_corpora_dict = None
        self._sihan_post_processing = 'false'
        if lang == 'ar':
            self._java_class = 'edu.stanford.nlp.international.arabic.process.ArabicSegmenter'
            model = 'arabic-segmenter-atb+bn+arztrain.ser.gz'
        elif lang == 'zh':
            self._java_class = 'edu.stanford.nlp.ie.crf.CRFClassifier'
            model = 'pku.gz'
            self._sihan_post_processing = 'true'
            path_to_dict = 'dict-chris6.ser.gz'
            try:
                self._dict = find_file(path_to_dict, searchpath=search_path, url=_stanford_url, verbose=False, env_vars=('STANFORD_MODELS',))
            except LookupError as e:
                raise LookupError("Could not find '%s' (tried using env. variables STANFORD_MODELS and <STANFORD_SEGMENTER>/data/)" % path_to_dict) from e
            sihan_dir = './data/'
            try:
                path_to_sihan_dir = find_dir(sihan_dir, url=_stanford_url, verbose=False, env_vars=('STANFORD_SEGMENTER',))
                self._sihan_corpora_dict = os.path.join(path_to_sihan_dir, sihan_dir)
            except LookupError as e:
                raise LookupError("Could not find '%s' (tried using the STANFORD_SEGMENTER environment variable)" % sihan_dir) from e
        else:
            raise LookupError(f'Unsupported language {lang}')
        try:
            self._model = find_file(model, searchpath=search_path, url=_stanford_url, verbose=False, env_vars=('STANFORD_MODELS', 'STANFORD_SEGMENTER'))
        except LookupError as e:
            raise LookupError("Could not find '%s' (tried using env. variables STANFORD_MODELS and <STANFORD_SEGMENTER>/data/)" % model) from e

    def tokenize(self, s):
        if False:
            print('Hello World!')
        super().tokenize(s)

    def segment_file(self, input_file_path):
        if False:
            for i in range(10):
                print('nop')
        ' '
        cmd = [self._java_class, '-loadClassifier', self._model, '-keepAllWhitespaces', self._keep_whitespaces, '-textFile', input_file_path]
        if self._sihan_corpora_dict is not None:
            cmd.extend(['-serDictionary', self._dict, '-sighanCorporaDict', self._sihan_corpora_dict, '-sighanPostProcessing', self._sihan_post_processing])
        stdout = self._execute(cmd)
        return stdout

    def segment(self, tokens):
        if False:
            print('Hello World!')
        return self.segment_sents([tokens])

    def segment_sents(self, sentences):
        if False:
            while True:
                i = 10
        ' '
        encoding = self._encoding
        (_input_fh, self._input_file_path) = tempfile.mkstemp(text=True)
        _input_fh = os.fdopen(_input_fh, 'wb')
        _input = '\n'.join((' '.join(x) for x in sentences))
        if isinstance(_input, str) and encoding:
            _input = _input.encode(encoding)
        _input_fh.write(_input)
        _input_fh.close()
        cmd = [self._java_class, '-loadClassifier', self._model, '-keepAllWhitespaces', self._keep_whitespaces, '-textFile', self._input_file_path]
        if self._sihan_corpora_dict is not None:
            cmd.extend(['-serDictionary', self._dict, '-sighanCorporaDict', self._sihan_corpora_dict, '-sighanPostProcessing', self._sihan_post_processing])
        stdout = self._execute(cmd)
        os.unlink(self._input_file_path)
        return stdout

    def _execute(self, cmd, verbose=False):
        if False:
            for i in range(10):
                print('nop')
        encoding = self._encoding
        cmd.extend(['-inputEncoding', encoding])
        _options_cmd = self._options_cmd
        if _options_cmd:
            cmd.extend(['-options', self._options_cmd])
        default_options = ' '.join(_java_options)
        config_java(options=self.java_options, verbose=verbose)
        (stdout, _stderr) = java(cmd, classpath=self._stanford_jar, stdout=PIPE, stderr=PIPE)
        stdout = stdout.decode(encoding)
        config_java(options=default_options, verbose=False)
        return stdout