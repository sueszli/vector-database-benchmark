import inspect
import os
import subprocess
import sys
import tempfile
from nltk.data import ZipFilePathPointer
from nltk.internals import find_dir, find_file, find_jars_within_path
from nltk.parse.api import ParserI
from nltk.parse.dependencygraph import DependencyGraph
from nltk.parse.util import taggedsents_to_conll

def malt_regex_tagger():
    if False:
        return 10
    from nltk.tag import RegexpTagger
    _tagger = RegexpTagger([('\\.$', '.'), ('\\,$', ','), ('\\?$', '?'), ('\\($', '('), ('\\)$', ')'), ('\\[$', '['), ('\\]$', ']'), ('^-?[0-9]+(\\.[0-9]+)?$', 'CD'), ('(The|the|A|a|An|an)$', 'DT'), ('(He|he|She|she|It|it|I|me|Me|You|you)$', 'PRP'), ('(His|his|Her|her|Its|its)$', 'PRP$'), ('(my|Your|your|Yours|yours)$', 'PRP$'), ('(on|On|in|In|at|At|since|Since)$', 'IN'), ('(for|For|ago|Ago|before|Before)$', 'IN'), ('(till|Till|until|Until)$', 'IN'), ('(by|By|beside|Beside)$', 'IN'), ('(under|Under|below|Below)$', 'IN'), ('(over|Over|above|Above)$', 'IN'), ('(across|Across|through|Through)$', 'IN'), ('(into|Into|towards|Towards)$', 'IN'), ('(onto|Onto|from|From)$', 'IN'), ('.*able$', 'JJ'), ('.*ness$', 'NN'), ('.*ly$', 'RB'), ('.*s$', 'NNS'), ('.*ing$', 'VBG'), ('.*ed$', 'VBD'), ('.*', 'NN')])
    return _tagger.tag

def find_maltparser(parser_dirname):
    if False:
        i = 10
        return i + 15
    '\n    A module to find MaltParser .jar file and its dependencies.\n    '
    if os.path.exists(parser_dirname):
        _malt_dir = parser_dirname
    else:
        _malt_dir = find_dir(parser_dirname, env_vars=('MALT_PARSER',))
    malt_dependencies = ['', '', '']
    _malt_jars = set(find_jars_within_path(_malt_dir))
    _jars = {os.path.split(jar)[1] for jar in _malt_jars}
    malt_dependencies = {'log4j.jar', 'libsvm.jar', 'liblinear-1.8.jar'}
    assert malt_dependencies.issubset(_jars)
    assert any(filter(lambda i: i.startswith('maltparser-') and i.endswith('.jar'), _jars))
    return list(_malt_jars)

def find_malt_model(model_filename):
    if False:
        print('Hello World!')
    '\n    A module to find pre-trained MaltParser model.\n    '
    if model_filename is None:
        return 'malt_temp.mco'
    elif os.path.exists(model_filename):
        return model_filename
    else:
        return find_file(model_filename, env_vars=('MALT_MODEL',), verbose=False)

class MaltParser(ParserI):
    """
    A class for dependency parsing with MaltParser. The input is the paths to:
    - (optionally) a maltparser directory
    - (optionally) the path to a pre-trained MaltParser .mco model file
    - (optionally) the tagger to use for POS tagging before parsing
    - (optionally) additional Java arguments

    Example:
        >>> from nltk.parse import malt
        >>> # With MALT_PARSER and MALT_MODEL environment set.
        >>> mp = malt.MaltParser(model_filename='engmalt.linear-1.7.mco') # doctest: +SKIP
        >>> mp.parse_one('I shot an elephant in my pajamas .'.split()).tree() # doctest: +SKIP
        (shot I (elephant an) (in (pajamas my)) .)
        >>> # Without MALT_PARSER and MALT_MODEL environment.
        >>> mp = malt.MaltParser('/home/user/maltparser-1.9.2/', '/home/user/engmalt.linear-1.7.mco') # doctest: +SKIP
        >>> mp.parse_one('I shot an elephant in my pajamas .'.split()).tree() # doctest: +SKIP
        (shot I (elephant an) (in (pajamas my)) .)
    """

    def __init__(self, parser_dirname='', model_filename=None, tagger=None, additional_java_args=None):
        if False:
            print('Hello World!')
        "\n        An interface for parsing with the Malt Parser.\n\n        :param parser_dirname: The path to the maltparser directory that\n            contains the maltparser-1.x.jar\n        :type parser_dirname: str\n        :param model_filename: The name of the pre-trained model with .mco file\n            extension. If provided, training will not be required.\n            (see http://www.maltparser.org/mco/mco.html and\n            see http://www.patful.com/chalk/node/185)\n        :type model_filename: str\n        :param tagger: The tagger used to POS tag the raw string before\n            formatting to CONLL format. It should behave like `nltk.pos_tag`\n        :type tagger: function\n        :param additional_java_args: This is the additional Java arguments that\n            one can use when calling Maltparser, usually this is the heapsize\n            limits, e.g. `additional_java_args=['-Xmx1024m']`\n            (see https://goo.gl/mpDBvQ)\n        :type additional_java_args: list\n        "
        self.malt_jars = find_maltparser(parser_dirname)
        self.additional_java_args = additional_java_args if additional_java_args is not None else []
        self.model = find_malt_model(model_filename)
        self._trained = self.model != 'malt_temp.mco'
        self.working_dir = tempfile.gettempdir()
        self.tagger = tagger if tagger is not None else malt_regex_tagger()

    def parse_tagged_sents(self, sentences, verbose=False, top_relation_label='null'):
        if False:
            return 10
        '\n        Use MaltParser to parse multiple POS tagged sentences. Takes multiple\n        sentences where each sentence is a list of (word, tag) tuples.\n        The sentences must have already been tokenized and tagged.\n\n        :param sentences: Input sentences to parse\n        :type sentence: list(list(tuple(str, str)))\n        :return: iter(iter(``DependencyGraph``)) the dependency graph\n            representation of each sentence\n        '
        if not self._trained:
            raise Exception('Parser has not been trained. Call train() first.')
        with tempfile.NamedTemporaryFile(prefix='malt_input.conll.', dir=self.working_dir, mode='w', delete=False) as input_file:
            with tempfile.NamedTemporaryFile(prefix='malt_output.conll.', dir=self.working_dir, mode='w', delete=False) as output_file:
                for line in taggedsents_to_conll(sentences):
                    input_file.write(str(line))
                input_file.close()
                cmd = self.generate_malt_command(input_file.name, output_file.name, mode='parse')
                _current_path = os.getcwd()
                try:
                    os.chdir(os.path.split(self.model)[0])
                except:
                    pass
                ret = self._execute(cmd, verbose)
                os.chdir(_current_path)
                if ret != 0:
                    raise Exception('MaltParser parsing (%s) failed with exit code %d' % (' '.join(cmd), ret))
                with open(output_file.name) as infile:
                    for tree_str in infile.read().split('\n\n'):
                        yield iter([DependencyGraph(tree_str, top_relation_label=top_relation_label)])
        os.remove(input_file.name)
        os.remove(output_file.name)

    def parse_sents(self, sentences, verbose=False, top_relation_label='null'):
        if False:
            return 10
        "\n        Use MaltParser to parse multiple sentences.\n        Takes a list of sentences, where each sentence is a list of words.\n        Each sentence will be automatically tagged with this\n        MaltParser instance's tagger.\n\n        :param sentences: Input sentences to parse\n        :type sentence: list(list(str))\n        :return: iter(DependencyGraph)\n        "
        tagged_sentences = (self.tagger(sentence) for sentence in sentences)
        return self.parse_tagged_sents(tagged_sentences, verbose, top_relation_label=top_relation_label)

    def generate_malt_command(self, inputfilename, outputfilename=None, mode=None):
        if False:
            i = 10
            return i + 15
        '\n        This function generates the maltparser command use at the terminal.\n\n        :param inputfilename: path to the input file\n        :type inputfilename: str\n        :param outputfilename: path to the output file\n        :type outputfilename: str\n        '
        cmd = ['java']
        cmd += self.additional_java_args
        classpaths_separator = ';' if sys.platform.startswith('win') else ':'
        cmd += ['-cp', classpaths_separator.join(self.malt_jars)]
        cmd += ['org.maltparser.Malt']
        if os.path.exists(self.model):
            cmd += ['-c', os.path.split(self.model)[-1]]
        else:
            cmd += ['-c', self.model]
        cmd += ['-i', inputfilename]
        if mode == 'parse':
            cmd += ['-o', outputfilename]
        cmd += ['-m', mode]
        return cmd

    @staticmethod
    def _execute(cmd, verbose=False):
        if False:
            return 10
        output = None if verbose else subprocess.PIPE
        p = subprocess.Popen(cmd, stdout=output, stderr=output)
        return p.wait()

    def train(self, depgraphs, verbose=False):
        if False:
            print('Hello World!')
        '\n        Train MaltParser from a list of ``DependencyGraph`` objects\n\n        :param depgraphs: list of ``DependencyGraph`` objects for training input data\n        :type depgraphs: DependencyGraph\n        '
        with tempfile.NamedTemporaryFile(prefix='malt_train.conll.', dir=self.working_dir, mode='w', delete=False) as input_file:
            input_str = '\n'.join((dg.to_conll(10) for dg in depgraphs))
            input_file.write(str(input_str))
        self.train_from_file(input_file.name, verbose=verbose)
        os.remove(input_file.name)

    def train_from_file(self, conll_file, verbose=False):
        if False:
            return 10
        '\n        Train MaltParser from a file\n        :param conll_file: str for the filename of the training input data\n        :type conll_file: str\n        '
        if isinstance(conll_file, ZipFilePathPointer):
            with tempfile.NamedTemporaryFile(prefix='malt_train.conll.', dir=self.working_dir, mode='w', delete=False) as input_file:
                with conll_file.open() as conll_input_file:
                    conll_str = conll_input_file.read()
                    input_file.write(str(conll_str))
                return self.train_from_file(input_file.name, verbose=verbose)
        cmd = self.generate_malt_command(conll_file, mode='learn')
        ret = self._execute(cmd, verbose)
        if ret != 0:
            raise Exception('MaltParser training (%s) failed with exit code %d' % (' '.join(cmd), ret))
        self._trained = True
if __name__ == '__main__':
    '\n    A demonstration function to show how NLTK users can use the malt parser API.\n\n    >>> from nltk import pos_tag\n    >>> assert \'MALT_PARSER\' in os.environ, str(\n    ... "Please set MALT_PARSER in your global environment, e.g.:\n"\n    ... "$ export MALT_PARSER=\'/home/user/maltparser-1.9.2/\'")\n    >>>\n    >>> assert \'MALT_MODEL\' in os.environ, str(\n    ... "Please set MALT_MODEL in your global environment, e.g.:\n"\n    ... "$ export MALT_MODEL=\'/home/user/engmalt.linear-1.7.mco\'")\n    >>>\n    >>> _dg1_str = str("1    John    _    NNP   _    _    2    SUBJ    _    _\n"\n    ...             "2    sees    _    VB    _    _    0    ROOT    _    _\n"\n    ...             "3    a       _    DT    _    _    4    SPEC    _    _\n"\n    ...             "4    dog     _    NN    _    _    2    OBJ     _    _\n"\n    ...             "5    .     _    .    _    _    2    PUNCT     _    _\n")\n    >>>\n    >>>\n    >>> _dg2_str  = str("1    John    _    NNP   _    _    2    SUBJ    _    _\n"\n    ...             "2    walks   _    VB    _    _    0    ROOT    _    _\n"\n    ...             "3    .     _    .    _    _    2    PUNCT     _    _\n")\n    >>> dg1 = DependencyGraph(_dg1_str)\n    >>> dg2 = DependencyGraph(_dg2_str)\n    >>> # Initialize a MaltParser object\n    >>> mp = MaltParser()\n    >>>\n    >>> # Trains a model.\n    >>> mp.train([dg1,dg2], verbose=False)\n    >>> sent1 = [\'John\',\'sees\',\'Mary\', \'.\']\n    >>> sent2 = [\'John\', \'walks\', \'a\', \'dog\', \'.\']\n    >>>\n    >>> # Parse a single sentence.\n    >>> parsed_sent1 = mp.parse_one(sent1)\n    >>> parsed_sent2 = mp.parse_one(sent2)\n    >>> print(parsed_sent1.tree())\n    (sees John Mary .)\n    >>> print(parsed_sent2.tree())\n    (walks John (dog a) .)\n    >>>\n    >>> # Parsing multiple sentences.\n    >>> sentences = [sent1,sent2]\n    >>> parsed_sents = mp.parse_sents(sentences)\n    >>> print(next(next(parsed_sents)).tree())\n    (sees John Mary .)\n    >>> print(next(next(parsed_sents)).tree())\n    (walks John (dog a) .)\n    >>>\n    >>> # Initialize a MaltParser object with an English pre-trained model.\n    >>> parser_dirname = \'maltparser-1.9.2\'\n    >>> model_name = \'engmalt.linear-1.7.mco\'\n    >>> mp = MaltParser(parser_dirname=parser_dirname, model_filename=model_name, tagger=pos_tag)\n    >>> sent1 = \'I shot an elephant in my pajamas .\'.split()\n    >>> sent2 = \'Time flies like banana .\'.split()\n    >>> # Parse a single sentence.\n    >>> print(mp.parse_one(sent1).tree())\n    (shot I (elephant an) (in (pajamas my)) .)\n    # Parsing multiple sentences\n    >>> sentences = [sent1,sent2]\n    >>> parsed_sents = mp.parse_sents(sentences)\n    >>> print(next(next(parsed_sents)).tree())\n    (shot I (elephant an) (in (pajamas my)) .)\n    >>> print(next(next(parsed_sents)).tree())\n    (flies Time (like banana) .)\n    '
    import doctest
    doctest.testmod()