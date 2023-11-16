import itertools
import re
from optparse import OptionParser
from bigdl.dllib.feature.dataset import base
from bigdl.dllib.feature.dataset import sentence
from bigdl.dllib.nn.layer import *
from bigdl.dllib.nn.criterion import *
from bigdl.dllib.optim.optimizer import *
from bigdl.dllib.utils.common import *
from bigdl.dllib.utils.common import Sample

def download_data(dest_dir):
    if False:
        return 10
    TINYSHAKESPEARE_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    file_name = 'input.txt'
    file_abs_path = base.maybe_download(file_name, dest_dir, TINYSHAKESPEARE_URL)
    return file_abs_path

def prepare_data(sc, folder, vocabsize, training_split):
    if False:
        return 10
    if not folder.startswith('hdfs://'):
        file = download_data(folder)
    else:
        file = folder
    sentences_rdd = sc.textFile(file).map(lambda line: sentence.sentences_split(line))
    pad_sent = sentences_rdd.flatMap(lambda x: x).map(lambda sent: sentence.sentences_bipadding(sent))
    tokens = pad_sent.map(lambda pad: sentence.sentence_tokenizer(pad))
    (train_tokens, val_tokens) = tokens.randomSplit([training_split, 1 - training_split])
    train_tokens.cache()
    val_tokens.cache()
    train_max_len = train_tokens.map(lambda x: len(x)).max()
    print('max length %s' % train_max_len)
    words = train_tokens.flatMap(lambda x: x)
    print('%s words and %s sentences processed in train data' % (words.count(), train_tokens.count()))
    val_max_len = val_tokens.map(lambda x: len(x)).max()
    print('val max length %s' % val_max_len)
    val_words = val_tokens.flatMap(lambda x: x)
    print('%s words and %s sentences processed in validation data' % (val_words.count(), val_tokens.count()))
    sort_words = words.map(lambda w: (w, 1)).reduceByKey(lambda a, b: a + b).sortBy(lambda w_c: w_c[1])
    vocabulary = np.array(sort_words.map(lambda w: w[0]).collect())
    fre_len = vocabulary.size
    if vocabsize > fre_len:
        length = fre_len
    else:
        length = vocabsize
    discard_vocab = vocabulary[:fre_len - length]
    used_vocab = vocabulary[fre_len - length:fre_len]
    used_vocab_size = used_vocab.size
    index = np.arange(used_vocab_size)
    index2word = dict(enumerate(used_vocab))
    word2index = dict(zip(used_vocab, index))
    total_vocab_len = used_vocab_size + 1
    startIdx = word2index.get('SENTENCESTART')
    endIdx = word2index.get('SENTENCEEND')

    def text2labeled(sent):
        if False:
            while True:
                i = 10
        indexes = [word2index.get(x, used_vocab_size) for x in sent]
        data = indexes[0:-1]
        label = indexes[1:len(indexes)]
        return (data, label)

    def labeled2onehotformat(labeled_sent):
        if False:
            i = 10
            return i + 15
        label = [x + 1 for x in labeled_sent[1]]
        size = len(labeled_sent[0])
        feature_onehot = np.zeros(size * total_vocab_len, dtype='int').reshape([size, total_vocab_len])
        for (i, el) in enumerate(labeled_sent[0]):
            feature_onehot[i, el] = 1
        return (feature_onehot, label)

    def padding(features, label, length):
        if False:
            while True:
                i = 10
        pad_len = length - len(label)
        padded_label = (label + [startIdx] * length)[:length]
        feature_padding = np.zeros((pad_len, total_vocab_len), dtype=np.int32)
        feature_padding[:, endIdx + 1] = np.ones(pad_len)
        padded_feautres = np.concatenate((features, feature_padding), axis=0)
        return (padded_feautres, padded_label)
    sample_rdd = train_tokens.map(lambda sentence_te: text2labeled(sentence_te)).map(lambda labeled_sent: labeled2onehotformat(labeled_sent)).map(lambda x: padding(x[0], x[1], train_max_len)).map(lambda vectors_label: Sample.from_ndarray(vectors_label[0], np.array(vectors_label[1]))).cache()
    val_sample_rdd = val_tokens.map(lambda sentence_t: text2labeled(sentence_t)).map(lambda labeled_sent: labeled2onehotformat(labeled_sent)).map(lambda x: padding(x[0], x[1], val_max_len)).map(lambda vectors_label: Sample.from_ndarray(vectors_label[0], np.array(vectors_label[1]))).cache()
    return (sample_rdd, val_sample_rdd, total_vocab_len)

def build_model(input_size, hidden_size, output_size, model_type):
    if False:
        print('Hello World!')
    if model_type == 'rnn':
        model = Sequential()
        model.add(Recurrent().add(RnnCell(input_size, hidden_size, Tanh()))).add(TimeDistributed(Linear(hidden_size, output_size)))
        model.reset()
        if get_bigdl_engine_type() == 'MklDnn':
            invalidInputError(False, 'Simple RNN is unsupported with MKL-DNN backend')
    elif model_type == 'lstm':
        model = Sequential()
        model.add(Recurrent().add(LSTM(input_size, hidden_size))).add(TimeDistributed(Linear(hidden_size, output_size)))
        model.reset()
        if get_bigdl_engine_type() == 'MklDnn':
            model = model.to_graph()
            model.set_input_formats([27])
            model.set_output_formats([27])
    return model
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-f', '--folder', dest='folder', default='/tmp/rnn')
    parser.add_option('-b', '--batchSize', dest='batchSize', default='12')
    parser.add_option('--learningRate', dest='learningrate', default='0.1')
    parser.add_option('--momentum', dest='momentum', default='0.0')
    parser.add_option('--weightDecay', dest='weight_decay', default='0.0')
    parser.add_option('--dampening', dest='dampening', default='0.0')
    parser.add_option('--hiddenSize', dest='hidden_size', default='40')
    parser.add_option('--vocabSize', dest='vob_size', default='4000')
    parser.add_option('--maxEpoch', dest='max_epoch', default='30')
    parser.add_option('--modelType', dest='model_type', default='rnn')
    (options, args) = parser.parse_args(sys.argv)
    batch_size = int(options.batchSize)
    learningrate = float(options.learningrate)
    momentum = float(options.momentum)
    weight_decay = float(options.weight_decay)
    dampening = float(options.dampening)
    hidden_size = int(options.hidden_size)
    vob_size = int(options.vob_size)
    max_epoch = int(options.max_epoch)
    folder = options.folder
    training_split = 0.8
    model_type = str(options.model_type)
    sc = SparkContext(appName='simplernn_example', conf=create_spark_conf())
    redire_spark_logs()
    show_bigdl_info_logs()
    init_engine()
    (train_rdd, val_rdd, vob_size) = prepare_data(sc, folder, vob_size, training_split)
    optimizer = Optimizer(model=build_model(vob_size, hidden_size, vob_size, model_type), training_rdd=train_rdd, criterion=TimeDistributedCriterion(CrossEntropyCriterion(), size_average=True), batch_size=batch_size, optim_method=SGD(learningrate=learningrate, weightdecay=weight_decay, momentum=momentum, dampening=dampening), end_trigger=MaxEpoch(max_epoch))
    optimizer.set_validation(batch_size=batch_size, val_rdd=val_rdd, trigger=EveryEpoch(), val_method=[Loss(TimeDistributedCriterion(CrossEntropyCriterion(), size_average=True))])
    train_model = optimizer.optimize()
    sc.stop()