"""Script to compute official BLEU score.

Source:
https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/bleu_hook.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
import sys
import unicodedata
import six
from absl import app as absl_app
from absl import flags
import tensorflow as tf
from official.transformer.utils import metrics
from official.transformer.utils import tokenizer
from official.utils.flags import core as flags_core

class UnicodeRegex(object):
    """Ad-hoc hack to recognize all punctuation and symbols."""

    def __init__(self):
        if False:
            return 10
        punctuation = self.property_chars('P')
        self.nondigit_punct_re = re.compile('([^\\d])([' + punctuation + '])')
        self.punct_nondigit_re = re.compile('([' + punctuation + '])([^\\d])')
        self.symbol_re = re.compile('([' + self.property_chars('S') + '])')

    def property_chars(self, prefix):
        if False:
            return 10
        return ''.join((six.unichr(x) for x in range(sys.maxunicode) if unicodedata.category(six.unichr(x)).startswith(prefix)))
uregex = UnicodeRegex()

def bleu_tokenize(string):
    if False:
        for i in range(10):
            print('nop')
    "Tokenize a string following the official BLEU implementation.\n\n  See https://github.com/moses-smt/mosesdecoder/'\n           'blob/master/scripts/generic/mteval-v14.pl#L954-L983\n  In our case, the input string is expected to be just one line\n  and no HTML entities de-escaping is needed.\n  So we just tokenize on punctuation and symbols,\n  except when a punctuation is preceded and followed by a digit\n  (e.g. a comma/dot as a thousand/decimal separator).\n\n  Note that a numer (e.g. a year) followed by a dot at the end of sentence\n  is NOT tokenized,\n  i.e. the dot stays with the number because `s/(\\p{P})(\\P{N})/ $1 $2/g`\n  does not match this case (unless we add a space after each sentence).\n  However, this error is already in the original mteval-v14.pl\n  and we want to be consistent with it.\n\n  Args:\n    string: the input string\n\n  Returns:\n    a list of tokens\n  "
    string = uregex.nondigit_punct_re.sub('\\1 \\2 ', string)
    string = uregex.punct_nondigit_re.sub(' \\1 \\2', string)
    string = uregex.symbol_re.sub(' \\1 ', string)
    return string.split()

def bleu_wrapper(ref_filename, hyp_filename, case_sensitive=False):
    if False:
        print('Hello World!')
    'Compute BLEU for two files (reference and hypothesis translation).'
    ref_lines = tokenizer.native_to_unicode(tf.io.gfile.GFile(ref_filename).read()).strip().splitlines()
    hyp_lines = tokenizer.native_to_unicode(tf.io.gfile.GFile(hyp_filename).read()).strip().splitlines()
    if len(ref_lines) != len(hyp_lines):
        raise ValueError('Reference and translation files have different number of lines. If training only a few steps (100-200), the translation may be empty.')
    if not case_sensitive:
        ref_lines = [x.lower() for x in ref_lines]
        hyp_lines = [x.lower() for x in hyp_lines]
    ref_tokens = [bleu_tokenize(x) for x in ref_lines]
    hyp_tokens = [bleu_tokenize(x) for x in hyp_lines]
    return metrics.compute_bleu(ref_tokens, hyp_tokens) * 100

def main(unused_argv):
    if False:
        i = 10
        return i + 15
    if FLAGS.bleu_variant in ('both', 'uncased'):
        score = bleu_wrapper(FLAGS.reference, FLAGS.translation, False)
        tf.logging.info('Case-insensitive results: %f' % score)
    if FLAGS.bleu_variant in ('both', 'cased'):
        score = bleu_wrapper(FLAGS.reference, FLAGS.translation, True)
        tf.logging.info('Case-sensitive results: %f' % score)

def define_compute_bleu_flags():
    if False:
        i = 10
        return i + 15
    'Add flags for computing BLEU score.'
    flags.DEFINE_string(name='translation', default=None, help=flags_core.help_wrap('File containing translated text.'))
    flags.mark_flag_as_required('translation')
    flags.DEFINE_string(name='reference', default=None, help=flags_core.help_wrap('File containing reference translation.'))
    flags.mark_flag_as_required('reference')
    flags.DEFINE_enum(name='bleu_variant', short_name='bv', default='both', enum_values=['both', 'uncased', 'cased'], case_sensitive=False, help=flags_core.help_wrap('Specify one or more BLEU variants to calculate. Variants: "cased", "uncased", or "both".'))
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    define_compute_bleu_flags()
    FLAGS = flags.FLAGS
    absl_app.run(main)