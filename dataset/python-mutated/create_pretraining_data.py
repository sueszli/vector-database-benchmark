"""Create masked LM/next sentence masked_lm TF examples for BERT."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import random
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
from official.nlp.bert import tokenization
FLAGS = flags.FLAGS
flags.DEFINE_string('input_file', None, 'Input raw text file (or comma-separated list of files).')
flags.DEFINE_string('output_file', None, 'Output TF example file (or comma-separated list of files).')
flags.DEFINE_string('vocab_file', None, 'The vocabulary file that the BERT model was trained on.')
flags.DEFINE_bool('do_lower_case', True, 'Whether to lower case the input text. Should be True for uncased models and False for cased models.')
flags.DEFINE_bool('do_whole_word_mask', False, 'Whether to use whole word masking rather than per-WordPiece masking.')
flags.DEFINE_bool('gzip_compress', False, 'Whether to use `GZIP` compress option to get compressed TFRecord files.')
flags.DEFINE_integer('max_seq_length', 128, 'Maximum sequence length.')
flags.DEFINE_integer('max_predictions_per_seq', 20, 'Maximum number of masked LM predictions per sequence.')
flags.DEFINE_integer('random_seed', 12345, 'Random seed for data generation.')
flags.DEFINE_integer('dupe_factor', 10, 'Number of times to duplicate the input data (with different masks).')
flags.DEFINE_float('masked_lm_prob', 0.15, 'Masked LM probability.')
flags.DEFINE_float('short_seq_prob', 0.1, 'Probability of creating sequences which are shorter than the maximum length.')

class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels, is_random_next):
        if False:
            print('Hello World!')
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.is_random_next = is_random_next
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

    def __str__(self):
        if False:
            while True:
                i = 10
        s = ''
        s += 'tokens: %s\n' % ' '.join([tokenization.printable_text(x) for x in self.tokens])
        s += 'segment_ids: %s\n' % ' '.join([str(x) for x in self.segment_ids])
        s += 'is_random_next: %s\n' % self.is_random_next
        s += 'masked_lm_positions: %s\n' % ' '.join([str(x) for x in self.masked_lm_positions])
        s += 'masked_lm_labels: %s\n' % ' '.join([tokenization.printable_text(x) for x in self.masked_lm_labels])
        s += '\n'
        return s

    def __repr__(self):
        if False:
            while True:
                i = 10
        return self.__str__()

def write_instance_to_example_files(instances, tokenizer, max_seq_length, max_predictions_per_seq, output_files):
    if False:
        while True:
            i = 10
    'Create TF example files from `TrainingInstance`s.'
    writers = []
    for output_file in output_files:
        writers.append(tf.io.TFRecordWriter(output_file, options='GZIP' if FLAGS.gzip_compress else ''))
    writer_index = 0
    total_written = 0
    for (inst_index, instance) in enumerate(instances):
        input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = list(instance.segment_ids)
        assert len(input_ids) <= max_seq_length
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_ids)
        while len(masked_lm_positions) < max_predictions_per_seq:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)
            masked_lm_weights.append(0.0)
        next_sentence_label = 1 if instance.is_random_next else 0
        features = collections.OrderedDict()
        features['input_ids'] = create_int_feature(input_ids)
        features['input_mask'] = create_int_feature(input_mask)
        features['segment_ids'] = create_int_feature(segment_ids)
        features['masked_lm_positions'] = create_int_feature(masked_lm_positions)
        features['masked_lm_ids'] = create_int_feature(masked_lm_ids)
        features['masked_lm_weights'] = create_float_feature(masked_lm_weights)
        features['next_sentence_labels'] = create_int_feature([next_sentence_label])
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)
        total_written += 1
        if inst_index < 20:
            logging.info('*** Example ***')
            logging.info('tokens: %s', ' '.join([tokenization.printable_text(x) for x in instance.tokens]))
            for feature_name in features.keys():
                feature = features[feature_name]
                values = []
                if feature.int64_list.value:
                    values = feature.int64_list.value
                elif feature.float_list.value:
                    values = feature.float_list.value
                logging.info('%s: %s', feature_name, ' '.join([str(x) for x in values]))
    for writer in writers:
        writer.close()
    logging.info('Wrote %d total instances', total_written)

def create_int_feature(values):
    if False:
        i = 10
        return i + 15
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature

def create_float_feature(values):
    if False:
        while True:
            i = 10
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature

def create_training_instances(input_files, tokenizer, max_seq_length, dupe_factor, short_seq_prob, masked_lm_prob, max_predictions_per_seq, rng):
    if False:
        i = 10
        return i + 15
    'Create `TrainingInstance`s from raw text.'
    all_documents = [[]]
    for input_file in input_files:
        with tf.io.gfile.GFile(input_file, 'rb') as reader:
            while True:
                line = tokenization.convert_to_unicode(reader.readline())
                if not line:
                    break
                line = line.strip()
                if not line:
                    all_documents.append([])
                tokens = tokenizer.tokenize(line)
                if tokens:
                    all_documents[-1].append(tokens)
    all_documents = [x for x in all_documents if x]
    rng.shuffle(all_documents)
    vocab_words = list(tokenizer.vocab.keys())
    instances = []
    for _ in range(dupe_factor):
        for document_index in range(len(all_documents)):
            instances.extend(create_instances_from_document(all_documents, document_index, max_seq_length, short_seq_prob, masked_lm_prob, max_predictions_per_seq, vocab_words, rng))
    rng.shuffle(instances)
    return instances

def create_instances_from_document(all_documents, document_index, max_seq_length, short_seq_prob, masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
    if False:
        print('Hello World!')
    'Creates `TrainingInstance`s for a single document.'
    document = all_documents[document_index]
    max_num_tokens = max_seq_length - 3
    target_seq_length = max_num_tokens
    if rng.random() < short_seq_prob:
        target_seq_length = rng.randint(2, max_num_tokens)
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = rng.randint(1, len(current_chunk) - 1)
                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])
                tokens_b = []
                is_random_next = False
                if len(current_chunk) == 1 or rng.random() < 0.5:
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)
                    for _ in range(10):
                        random_document_index = rng.randint(0, len(all_documents) - 1)
                        if random_document_index != document_index:
                            break
                    random_document = all_documents[random_document_index]
                    random_start = rng.randint(0, len(random_document) - 1)
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)
                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1
                tokens = []
                segment_ids = []
                tokens.append('[CLS]')
                segment_ids.append(0)
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)
                tokens.append('[SEP]')
                segment_ids.append(0)
                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append('[SEP]')
                segment_ids.append(1)
                (tokens, masked_lm_positions, masked_lm_labels) = create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
                instance = TrainingInstance(tokens=tokens, segment_ids=segment_ids, is_random_next=is_random_next, masked_lm_positions=masked_lm_positions, masked_lm_labels=masked_lm_labels)
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1
    return instances
MaskedLmInstance = collections.namedtuple('MaskedLmInstance', ['index', 'label'])

def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
    if False:
        for i in range(10):
            print('nop')
    'Creates the predictions for the masked LM objective.'
    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == '[CLS]' or token == '[SEP]':
            continue
        if FLAGS.do_whole_word_mask and len(cand_indexes) >= 1 and token.startswith('##'):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])
    rng.shuffle(cand_indexes)
    output_tokens = list(tokens)
    num_to_predict = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))
    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)
            masked_token = None
            if rng.random() < 0.8:
                masked_token = '[MASK]'
            elif rng.random() < 0.5:
                masked_token = tokens[index]
            else:
                masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]
            output_tokens[index] = masked_token
            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)
    return (output_tokens, masked_lm_positions, masked_lm_labels)

def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    if False:
        return 10
    'Truncates a pair of sequences to a maximum sequence length.'
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break
        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1
        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()

def main(_):
    if False:
        i = 10
        return i + 15
    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    input_files = []
    for input_pattern in FLAGS.input_file.split(','):
        input_files.extend(tf.io.gfile.glob(input_pattern))
    logging.info('*** Reading from input files ***')
    for input_file in input_files:
        logging.info('  %s', input_file)
    rng = random.Random(FLAGS.random_seed)
    instances = create_training_instances(input_files, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor, FLAGS.short_seq_prob, FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq, rng)
    output_files = FLAGS.output_file.split(',')
    logging.info('*** Writing to output files ***')
    for output_file in output_files:
        logging.info('  %s', output_file)
    write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length, FLAGS.max_predictions_per_seq, output_files)
if __name__ == '__main__':
    flags.mark_flag_as_required('input_file')
    flags.mark_flag_as_required('output_file')
    flags.mark_flag_as_required('vocab_file')
    app.run(main)