import os
from PIL import Image
import numpy as np
import tensorflow as tf
data_dir = 'G:/Deeplearn/OwnCollection/OwnCollection/non-vehicles/Test2'
re_dir = 'G:/Deeplearn/OwnCollection/OwnCollection/non-vehicles/Test'
train = False
model_path = 'model/image_model'

def read_data(data_dir, re_dir):
    if False:
        i = 10
        return i + 15
    datas = []
    labels = []
    fpaths = []
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        fpaths.append(fpath)
        print(fpath)
        image = Image.open(fpath)
        data = np.array(image) / 255.0
        label = 0
        datas.append(data)
        labels.append(label)
        print(data.shape)
    for fname in os.listdir(re_dir):
        fpath = os.path.join(re_dir, fname)
        print(fpath)
        fpaths.append(fpath)
        image = Image.open(fpath)
        data = np.array(image) / 255.0
        label = 1
        datas.append(data)
        labels.append(label)
        print(data.shape)
    datas = np.array(datas)
    labels = np.array(labels)
    print('shape of datas: {}\tshape of labels: {}'.format(datas.shape, labels.shape))
    return (fpaths, datas, labels)
(fpaths, datas, labels) = read_data(data_dir, re_dir)
num_classes = len(set(labels))
datas_placeholder = tf.placeholder(tf.float32, [None, 64, 64, 3])
labels_placeholder = tf.placeholder(tf.int32, [None])
dropout_placeholdr = tf.placeholder(tf.float32)
conv0 = tf.layers.conv2d(datas_placeholder, 20, 5, activation=tf.nn.relu)
pool0 = tf.layers.max_pooling2d(conv0, [2, 2], [2, 2])
conv1 = tf.layers.conv2d(pool0, 40, 4, activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(conv1, [2, 2], [2, 2])
conv2 = tf.layers.conv2d(pool0, 60, 3, activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(conv2, [2, 2], [2, 2])
flatten = tf.layers.flatten(pool2)
fc = tf.layers.dense(flatten, 400, activation=tf.nn.relu)
dropout_fc = tf.layers.dropout(fc, dropout_placeholdr)
logits = tf.layers.dense(dropout_fc, num_classes)
predicted_labels = tf.arg_max(logits, 1)
losses = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(labels_placeholder, num_classes), logits=logits)
mean_loss = tf.reduce_mean(losses)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(losses)
saver = tf.train.Saver()
with tf.Session() as sess:
    if train:
        print('训练模式')
        sess.run(tf.global_variables_initializer())
        train_feed_dict = {datas_placeholder: datas, labels_placeholder: labels, dropout_placeholdr: 0.5}
        for step in range(150):
            (_, mean_loss_val) = sess.run([optimizer, mean_loss], feed_dict=train_feed_dict)
            if step % 10 == 0:
                print('step = {}\tmean loss = {}'.format(step, mean_loss_val))
        saver.save(sess, model_path)
        print('训练结束，保存模型到{}'.format(model_path))
    else:
        print('测试模式')
        saver.restore(sess, model_path)
        print('从{}载入模型'.format(model_path))
        label_name_dict = {0: '巴士', 1: '出租车', 2: '货车', 3: '家用轿车', 4: '面包车', 5: '吉普车', 6: '运动型多功能车', 7: '重型货车', 8: '赛车', 9: '消防车'}
        test_feed_dict = {datas_placeholder: datas, labels_placeholder: labels, dropout_placeholdr: 0}
        predicted_labels_val = sess.run(predicted_labels, feed_dict=test_feed_dict)
        for (fpath, real_label, predicted_label) in zip(fpaths, labels, predicted_labels_val):
            real_label_name = label_name_dict[real_label]
            predicted_label_name = label_name_dict[predicted_label]
            print('{}\t{} => {}'.format(fpath, real_label_name, predicted_label_name))
        correct_number = 0
        for (fpath, real_label, predicted_label) in zip(fpaths, labels, predicted_labels_val):
            if real_label == predicted_label:
                correct_number += 1
        correct_rate = correct_number / 200
        print('正确率: {:.2%}'.format(correct_rate))