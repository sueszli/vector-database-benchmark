"""Some simple tools for error counting.

"""
import collections
ErrorCounts = collections.namedtuple('ErrorCounts', ['fn', 'fp', 'truth_count', 'test_count'])
ErrorRates = collections.namedtuple('ErrorRates', ['label_error', 'word_recall_error', 'word_precision_error', 'sequence_error'])

def CountWordErrors(ocr_text, truth_text):
    if False:
        i = 10
        return i + 15
    'Counts the word drop and add errors as a bag of words.\n\n  Args:\n    ocr_text:    OCR text string.\n    truth_text:  Truth text string.\n\n  Returns:\n    ErrorCounts named tuple.\n  '
    return CountErrors(ocr_text.split(), truth_text.split())

def CountErrors(ocr_text, truth_text):
    if False:
        i = 10
        return i + 15
    'Counts the drops and adds between 2 bags of iterables.\n\n  Simple bag of objects count returns the number of dropped and added\n  elements, regardless of order, from anything that is iterable, eg\n  a pair of strings gives character errors, and a pair of word lists give\n  word errors.\n  Args:\n    ocr_text:    OCR text iterable (eg string for chars, word list for words).\n    truth_text:  Truth text iterable.\n\n  Returns:\n    ErrorCounts named tuple.\n  '
    counts = collections.Counter(truth_text)
    counts.subtract(ocr_text)
    drops = sum((c for c in counts.values() if c > 0))
    adds = sum((-c for c in counts.values() if c < 0))
    return ErrorCounts(drops, adds, len(truth_text), len(ocr_text))

def AddErrors(counts1, counts2):
    if False:
        while True:
            i = 10
    'Adds the counts and returns a new sum tuple.\n\n  Args:\n    counts1: ErrorCounts named tuples to sum.\n    counts2: ErrorCounts named tuples to sum.\n  Returns:\n    Sum of counts1, counts2.\n  '
    return ErrorCounts(counts1.fn + counts2.fn, counts1.fp + counts2.fp, counts1.truth_count + counts2.truth_count, counts1.test_count + counts2.test_count)

def ComputeErrorRates(label_counts, word_counts, seq_errors, num_seqs):
    if False:
        return 10
    'Returns an ErrorRates corresponding to the given counts.\n\n  Args:\n    label_counts: ErrorCounts for the character labels\n    word_counts:  ErrorCounts for the words\n    seq_errors:   Number of sequence errors\n    num_seqs:     Total sequences\n  Returns:\n    ErrorRates corresponding to the given counts.\n  '
    label_errors = label_counts.fn + label_counts.fp
    num_labels = label_counts.truth_count + label_counts.test_count
    return ErrorRates(ComputeErrorRate(label_errors, num_labels), ComputeErrorRate(word_counts.fn, word_counts.truth_count), ComputeErrorRate(word_counts.fp, word_counts.test_count), ComputeErrorRate(seq_errors, num_seqs))

def ComputeErrorRate(error_count, truth_count):
    if False:
        while True:
            i = 10
    'Returns a sanitized percent error rate from the raw counts.\n\n  Prevents div by 0 and clips return to 100%.\n  Args:\n    error_count: Number of errors.\n    truth_count: Number to divide by.\n\n  Returns:\n    100.0 * error_count / truth_count clipped to 100.\n  '
    if truth_count == 0:
        truth_count = 1
        error_count = 1
    elif error_count > truth_count:
        error_count = truth_count
    return error_count * 100.0 / truth_count