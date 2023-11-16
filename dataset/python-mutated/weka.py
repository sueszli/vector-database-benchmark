"""
Classifiers that make use of the external 'Weka' package.
"""
import os
import re
import subprocess
import tempfile
import time
import zipfile
from sys import stdin
from nltk.classify.api import ClassifierI
from nltk.internals import config_java, java
from nltk.probability import DictionaryProbDist
_weka_classpath = None
_weka_search = ['.', '/usr/share/weka', '/usr/local/share/weka', '/usr/lib/weka', '/usr/local/lib/weka']

def config_weka(classpath=None):
    if False:
        while True:
            i = 10
    global _weka_classpath
    config_java()
    if classpath is not None:
        _weka_classpath = classpath
    if _weka_classpath is None:
        searchpath = _weka_search
        if 'WEKAHOME' in os.environ:
            searchpath.insert(0, os.environ['WEKAHOME'])
        for path in searchpath:
            if os.path.exists(os.path.join(path, 'weka.jar')):
                _weka_classpath = os.path.join(path, 'weka.jar')
                version = _check_weka_version(_weka_classpath)
                if version:
                    print(f'[Found Weka: {_weka_classpath} (version {version})]')
                else:
                    print('[Found Weka: %s]' % _weka_classpath)
                _check_weka_version(_weka_classpath)
    if _weka_classpath is None:
        raise LookupError('Unable to find weka.jar!  Use config_weka() or set the WEKAHOME environment variable. For more information about Weka, please see https://www.cs.waikato.ac.nz/ml/weka/')

def _check_weka_version(jar):
    if False:
        return 10
    try:
        zf = zipfile.ZipFile(jar)
    except (SystemExit, KeyboardInterrupt):
        raise
    except:
        return None
    try:
        try:
            return zf.read('weka/core/version.txt')
        except KeyError:
            return None
    finally:
        zf.close()

class WekaClassifier(ClassifierI):

    def __init__(self, formatter, model_filename):
        if False:
            i = 10
            return i + 15
        self._formatter = formatter
        self._model = model_filename

    def prob_classify_many(self, featuresets):
        if False:
            return 10
        return self._classify_many(featuresets, ['-p', '0', '-distribution'])

    def classify_many(self, featuresets):
        if False:
            while True:
                i = 10
        return self._classify_many(featuresets, ['-p', '0'])

    def _classify_many(self, featuresets, options):
        if False:
            return 10
        config_weka()
        temp_dir = tempfile.mkdtemp()
        try:
            test_filename = os.path.join(temp_dir, 'test.arff')
            self._formatter.write(test_filename, featuresets)
            cmd = ['weka.classifiers.bayes.NaiveBayes', '-l', self._model, '-T', test_filename] + options
            (stdout, stderr) = java(cmd, classpath=_weka_classpath, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if stderr and (not stdout):
                if 'Illegal options: -distribution' in stderr:
                    raise ValueError('The installed version of weka does not support probability distribution output.')
                else:
                    raise ValueError('Weka failed to generate output:\n%s' % stderr)
            return self.parse_weka_output(stdout.decode(stdin.encoding).split('\n'))
        finally:
            for f in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, f))
            os.rmdir(temp_dir)

    def parse_weka_distribution(self, s):
        if False:
            i = 10
            return i + 15
        probs = [float(v) for v in re.split('[*,]+', s) if v.strip()]
        probs = dict(zip(self._formatter.labels(), probs))
        return DictionaryProbDist(probs)

    def parse_weka_output(self, lines):
        if False:
            while True:
                i = 10
        for (i, line) in enumerate(lines):
            if line.strip().startswith('inst#'):
                lines = lines[i:]
                break
        if lines[0].split() == ['inst#', 'actual', 'predicted', 'error', 'prediction']:
            return [line.split()[2].split(':')[1] for line in lines[1:] if line.strip()]
        elif lines[0].split() == ['inst#', 'actual', 'predicted', 'error', 'distribution']:
            return [self.parse_weka_distribution(line.split()[-1]) for line in lines[1:] if line.strip()]
        elif re.match('^0 \\w+ [01]\\.[0-9]* \\?\\s*$', lines[0]):
            return [line.split()[1] for line in lines if line.strip()]
        else:
            for line in lines[:10]:
                print(line)
            raise ValueError('Unhandled output format -- your version of weka may not be supported.\n  Header: %s' % lines[0])
    _CLASSIFIER_CLASS = {'naivebayes': 'weka.classifiers.bayes.NaiveBayes', 'C4.5': 'weka.classifiers.trees.J48', 'log_regression': 'weka.classifiers.functions.Logistic', 'svm': 'weka.classifiers.functions.SMO', 'kstar': 'weka.classifiers.lazy.KStar', 'ripper': 'weka.classifiers.rules.JRip'}

    @classmethod
    def train(cls, model_filename, featuresets, classifier='naivebayes', options=[], quiet=True):
        if False:
            i = 10
            return i + 15
        config_weka()
        formatter = ARFF_Formatter.from_train(featuresets)
        temp_dir = tempfile.mkdtemp()
        try:
            train_filename = os.path.join(temp_dir, 'train.arff')
            formatter.write(train_filename, featuresets)
            if classifier in cls._CLASSIFIER_CLASS:
                javaclass = cls._CLASSIFIER_CLASS[classifier]
            elif classifier in cls._CLASSIFIER_CLASS.values():
                javaclass = classifier
            else:
                raise ValueError('Unknown classifier %s' % classifier)
            cmd = [javaclass, '-d', model_filename, '-t', train_filename]
            cmd += list(options)
            if quiet:
                stdout = subprocess.PIPE
            else:
                stdout = None
            java(cmd, classpath=_weka_classpath, stdout=stdout)
            return WekaClassifier(formatter, model_filename)
        finally:
            for f in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, f))
            os.rmdir(temp_dir)

class ARFF_Formatter:
    """
    Converts featuresets and labeled featuresets to ARFF-formatted
    strings, appropriate for input into Weka.

    Features and classes can be specified manually in the constructor, or may
    be determined from data using ``from_train``.
    """

    def __init__(self, labels, features):
        if False:
            while True:
                i = 10
        '\n        :param labels: A list of all class labels that can be generated.\n        :param features: A list of feature specifications, where\n            each feature specification is a tuple (fname, ftype);\n            and ftype is an ARFF type string such as NUMERIC or\n            STRING.\n        '
        self._labels = labels
        self._features = features

    def format(self, tokens):
        if False:
            i = 10
            return i + 15
        'Returns a string representation of ARFF output for the given data.'
        return self.header_section() + self.data_section(tokens)

    def labels(self):
        if False:
            print('Hello World!')
        'Returns the list of classes.'
        return list(self._labels)

    def write(self, outfile, tokens):
        if False:
            print('Hello World!')
        'Writes ARFF data to a file for the given data.'
        if not hasattr(outfile, 'write'):
            outfile = open(outfile, 'w')
        outfile.write(self.format(tokens))
        outfile.close()

    @staticmethod
    def from_train(tokens):
        if False:
            i = 10
            return i + 15
        '\n        Constructs an ARFF_Formatter instance with class labels and feature\n        types determined from the given data. Handles boolean, numeric and\n        string (note: not nominal) types.\n        '
        labels = {label for (tok, label) in tokens}
        features = {}
        for (tok, label) in tokens:
            for (fname, fval) in tok.items():
                if issubclass(type(fval), bool):
                    ftype = '{True, False}'
                elif issubclass(type(fval), (int, float, bool)):
                    ftype = 'NUMERIC'
                elif issubclass(type(fval), str):
                    ftype = 'STRING'
                elif fval is None:
                    continue
                else:
                    raise ValueError('Unsupported value type %r' % ftype)
                if features.get(fname, ftype) != ftype:
                    raise ValueError('Inconsistent type for %s' % fname)
                features[fname] = ftype
        features = sorted(features.items())
        return ARFF_Formatter(labels, features)

    def header_section(self):
        if False:
            while True:
                i = 10
        'Returns an ARFF header as a string.'
        s = '% Weka ARFF file\n' + '% Generated automatically by NLTK\n' + '%% %s\n\n' % time.ctime()
        s += '@RELATION rel\n\n'
        for (fname, ftype) in self._features:
            s += '@ATTRIBUTE %-30r %s\n' % (fname, ftype)
        s += '@ATTRIBUTE %-30r {%s}\n' % ('-label-', ','.join(self._labels))
        return s

    def data_section(self, tokens, labeled=None):
        if False:
            while True:
                i = 10
        "\n        Returns the ARFF data section for the given data.\n\n        :param tokens: a list of featuresets (dicts) or labelled featuresets\n            which are tuples (featureset, label).\n        :param labeled: Indicates whether the given tokens are labeled\n            or not.  If None, then the tokens will be assumed to be\n            labeled if the first token's value is a tuple or list.\n        "
        if labeled is None:
            labeled = tokens and isinstance(tokens[0], (tuple, list))
        if not labeled:
            tokens = [(tok, None) for tok in tokens]
        s = '\n@DATA\n'
        for (tok, label) in tokens:
            for (fname, ftype) in self._features:
                s += '%s,' % self._fmt_arff_val(tok.get(fname))
            s += '%s\n' % self._fmt_arff_val(label)
        return s

    def _fmt_arff_val(self, fval):
        if False:
            i = 10
            return i + 15
        if fval is None:
            return '?'
        elif isinstance(fval, (bool, int)):
            return '%s' % fval
        elif isinstance(fval, float):
            return '%r' % fval
        else:
            return '%r' % fval
if __name__ == '__main__':
    from nltk.classify.util import binary_names_demo_features, names_demo

    def make_classifier(featuresets):
        if False:
            print('Hello World!')
        return WekaClassifier.train('/tmp/name.model', featuresets, 'C4.5')
    classifier = names_demo(make_classifier, binary_names_demo_features)