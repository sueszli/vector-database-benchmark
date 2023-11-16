import third_party.rnn_cell as rnn_cell
import os
import sys
import numpy as np
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import array_ops
import tensorflow as tf
import data_utils
np.set_printoptions(threshold='nan')
(word2id, id2word, P_Emb, P_sig) = data_utils.Word2vec()

def read_data(source_path, type=0):
    if False:
        print('Hello World!')
    data_set = []
    with tf.gfile.GFile(source_path, mode='r') as source_file:
        charac_num = len(word2id)
        for line in source_file.readlines():
            if type == 0:
                (_, target) = line.split('==')
                lines = target.strip().split('\t')
                for l in lines:
                    target_ids = [word2id.get(x, charac_num - 1) for x in l.split(' ')]
                    data_set.append(target_ids)
            else:
                target_ids = [word2id.get(word.encode('utf-8'), charac_num - 1) for word in line.strip().decode('utf-8') if word != u'\ufeff']
                if len(target_ids) == 7:
                    data_set.append(target_ids)
    return data_set

def geneMemory(decoder_inputs, cell_initializer=tf.constant_initializer(np.array(P_Emb, dtype=np.float32)), size=500, embedding_size=200, output_keep_prob=1.0, initial_state=None, dtype=tf.float32):
    if False:
        i = 10
        return i + 15
    "\n    \tembedding_attention_seq2seq: embedding_attention_decoder相对应的seq2seq----------对encoder输入进行embedding，运行encoder部分，将encoder输出作为参数传给embedding_attention_decoder\n    \tembedding_attention_decoder:embedding_decoder和attention_decoder-----------对decoder_input进行embedding，定义loop_function，调用attention_decoder;\n    \t\t也就是说：embedding_attention_decoder第一步创建了解码用的embedding； 第二步创建了一个循环函数loop_function，用于将上一步的输出映射到词表空间，输出一个word embedding作为下一步的输入；最后是我们最关注的attention_decoder部分完成解码工作\n    \t\t疑难解析：1、output_size 与 num_symbols的差别：output_size是rnn的一个cell输出的大小，num_symbols是最终的输出大小，对应着词汇表的大小\n    \tattention_decoder: 所谓的attention，就是在每个解码的时间步，对encoder的隐层状态进行加权求和，针对不同信息进行不同程度的注意力。\n    \t\t\t\t\t\t那么我们的重点就是求出不同隐层状态对应的权重。源码中的attention机制里是最常见的一种，可以分为三步走：（1）通过当前隐层状态(d_{t})和关注的隐层状态(h_{i})求出对应权重u^{t}_{i}；（2）softmax归一化为概率；（3）作为加权系数对不同隐层状态求和，得到一个的信息向量d^{'}_{t}。后续的d^{'}_{t}使用会因为具体任务有所差别。\n    \t\tencoder输出的隐层状态(h_{1},...,h_{T_{A}}), decoder的隐层状态(d_{1},...,d_{T_{B}})\n    "
    with variable_scope.variable_scope('embedding_attention_seq2seq'):
        with variable_scope.variable_scope('embedding_attention_decoder'):
            with variable_scope.variable_scope('attention_decoder'):
                cell = rnn_cell.BasicLSTMCell(size, state_is_tuple=True)
                embedding = variable_scope.get_variable('embedding', [4777, embedding_size], initializer=cell_initializer, trainable=False)
                embed_decoder_inputs = [embedding_ops.embedding_lookup(embedding, i) for (index, i) in enumerate(decoder_inputs)]
                batch_size = embed_decoder_inputs[0].get_shape()[0].value
                if initial_state is not None:
                    state = initial_state
                else:
                    state = cell.zero_state(batch_size, dtype)
                weight_zero = array_ops.zeros(array_ops.pack([batch_size, 1500]), dtype=dtype)
                cell_outputs = []
                for (i, inp) in enumerate(embed_decoder_inputs):
                    if i > 0:
                        variable_scope.get_variable_scope().reuse_variables()
                    (cell_output, state) = cell(array_ops.concat(1, [inp, weight_zero]), state)
                    cell_outputs.append(cell_output)
    return cell_outputs

def MemNN(batch_size=1, length=7):
    if False:
        return 10
    decoder_inputs = []
    for i in xrange(length):
        decoder_inputs.append(tf.placeholder(tf.int32, shape=[batch_size], name='decoder{0}'.format(i)))
    outputs = geneMemory(decoder_inputs)
    return (decoder_inputs, outputs)

def createMemory():
    if False:
        for i in range(10):
            print('nop')
    print('Create memory files begin')
    options = data_utils.get_memory_options()
    if sys.argv[2] == 'biansai' or 'tianyuan' or 'yanqing' or 'other':
        memory_file = options[sys.argv[2] + '_file']
        file_type = options[sys.argv[2] + '_type']
    else:
        memory_file = options['general_file']
        file_type = options['general_type']
    data_set = read_data('../resource/memory_resource/text/' + memory_file, type=file_type)
    batch_size = 4
    if batch_size:
        data_set_temp = []
        for i_th in range(int(len(data_set) // batch_size)):
            data_temp = []
            for i in range(batch_size):
                if i == 0:
                    data_temp += [word2id['START']]
                if i != batch_size - 1:
                    data_temp += data_set[i_th * batch_size + i] + [word2id['/']]
                else:
                    data_temp += data_set[i_th * batch_size + i] + [word2id['/']] + data_set[i_th * batch_size] + [word2id['END']]
            data_set_temp.append(data_temp)
    data_set = data_set_temp
    with tf.Session() as sess:
        print('Create memory files: build the model')
        (decoder_inputs, output_feed) = MemNN(length=len(data_set[0]))
        path = os.getcwd() + '/model'
        list_file = [sys.argv[1]]
        for f in list_file:
            print('Create memory files: reading model parameters from %s' % f)
            sess.run(tf.initialize_all_variables())
            saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=200)
            saver.restore(sess, path + '/' + f)
            memory = [[]]
            memoryWordVector = []
            for line in data_set:
                input_feed = {}
                line_length = len(line)
                for l in xrange(line_length):
                    input_feed[decoder_inputs[l].name] = np.array([line[l]], dtype=np.int32)
                outputs = sess.run(output_feed, input_feed)
                for output in outputs[:-1]:
                    memory[0].append(output[0])
                for l in xrange(line_length):
                    if l == 0:
                        memoryWordVector.append(np.array([0.0] * 200, dtype=np.float32) + 1e-06)
                    elif l != line_length - 1:
                        memoryWordVector.append(np.array(P_Emb[line[l + 1]], dtype=np.float32))
            np.save('../resource/memory_resource/npy/' + sys.argv[1] + '_' + sys.argv[2] + '_memoryWordVector.npy', np.array(memoryWordVector, dtype=np.float32))
            np.save('../resource/memory_resource/npy/' + sys.argv[1] + '_' + sys.argv[2] + '_memory.npy', np.array(memory, dtype=np.float32))
            print('Create memory files done!')

def main():
    if False:
        for i in range(10):
            print('nop')
    createMemory()
if __name__ == '__main__':
    createMemory()