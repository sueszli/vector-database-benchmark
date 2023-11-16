"""TFRecord related utilities."""
from six.moves import range
import tensorflow as tf

def int64_feature(value):
    if False:
        return 10
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def int64_list_feature(value):
    if False:
        for i in range(10):
            print('nop')
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
    if False:
        while True:
            i = 10
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def bytes_list_feature(value):
    if False:
        return 10
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_list_feature(value):
    if False:
        print('Hello World!')
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def read_examples_list(path):
    if False:
        for i in range(10):
            print('nop')
    'Read list of training or validation examples.\n\n    The file is assumed to contain a single example per line where the first\n    token in the line is an identifier that allows us to find the image and\n    annotation xml for that example.\n\n    For example, the line:\n    xyz 3\n    would allow us to find files xyz.jpg and xyz.xml (the 3 would be ignored).\n\n    Args:\n      path: absolute path to examples list file.\n\n    Returns:\n      list of example identifiers (strings).\n    '
    with tf.io.gfile.GFile(path) as fid:
        lines = fid.readlines()
    return [line.strip().split(' ')[0] for line in lines]

def recursive_parse_xml_to_dict(xml):
    if False:
        return 10
    'Recursively parses XML contents to python dict.\n\n    We assume that `object` tags are the only ones that can appear\n    multiple times at the same level of a tree.\n\n    Args:\n      xml: xml tree obtained by parsing XML file contents using lxml.etree\n\n    Returns:\n      Python dictionary holding XML contents.\n    '
    if not xml:
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = recursive_parse_xml_to_dict(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}

def open_sharded_output_tfrecords(exit_stack, base_path, num_shards):
    if False:
        print('Hello World!')
    'Opens all TFRecord shards for writing and adds them to an exit stack.\n\n    Args:\n      exit_stack: A context2.ExitStack used to automatically closed the TFRecords\n        opened in this function.\n      base_path: The base path for all shards\n      num_shards: The number of shards\n\n    Returns:\n      The list of opened TFRecords. Position k in the list corresponds to shard k.\n    '
    tf_record_output_filenames = ['{}-{:05d}-of-{:05d}'.format(base_path, idx, num_shards) for idx in range(num_shards)]
    tfrecords = [exit_stack.enter_context(tf.io.TFRecordWriter(file_name)) for file_name in tf_record_output_filenames]
    return tfrecords