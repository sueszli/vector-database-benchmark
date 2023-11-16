from __future__ import print_function, division
from builtins import range, input
from keras.layers import Input, Lambda, Dense, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from collections import Counter
files = glob('../large_files/yalefaces/subject*')
np.random.shuffle(files)
N = len(files)

def load_img(filepath):
    if False:
        return 10
    img = image.img_to_array(image.load_img(filepath, target_size=[60, 80])).astype('uint8')
    return img
img = load_img(np.random.choice(files))
plt.imshow(img)
plt.show()
shape = [N] + list(img.shape)
images = np.zeros(shape)
for (i, f) in enumerate(files):
    img = load_img(f)
    images[i] = img
labels = np.zeros(N)
for (i, f) in enumerate(files):
    filename = f.rsplit('/', 1)[-1]
    subject_num = filename.split('.', 1)[0]
    idx = int(subject_num.replace('subject', '')) - 1
    labels[i] = idx
label_count = Counter(labels)
unique_labels = set(label_count.keys())
n_subjects = len(label_count)
n_test = 3 * n_subjects
n_train = N - n_test
train_images = np.zeros([n_train] + list(img.shape))
train_labels = np.zeros(n_train)
test_images = np.zeros([n_test] + list(img.shape))
test_labels = np.zeros(n_test)
count_so_far = {}
train_idx = 0
test_idx = 0
for (img, label) in zip(images, labels):
    count_so_far[label] = count_so_far.get(label, 0) + 1
    if count_so_far[label] > 3:
        train_images[train_idx] = img
        train_labels[train_idx] = label
        train_idx += 1
    else:
        test_images[test_idx] = img
        test_labels[test_idx] = label
        test_idx += 1
train_label2idx = {}
test_label2idx = {}
for (i, label) in enumerate(train_labels):
    if label not in train_label2idx:
        train_label2idx[label] = [i]
    else:
        train_label2idx[label].append(i)
for (i, label) in enumerate(test_labels):
    if label not in test_label2idx:
        test_label2idx[label] = [i]
    else:
        test_label2idx[label].append(i)
train_positives = []
train_negatives = []
test_positives = []
test_negatives = []
for (label, indices) in train_label2idx.items():
    other_indices = set(range(n_train)) - set(indices)
    for (i, idx1) in enumerate(indices):
        for idx2 in indices[i + 1:]:
            train_positives.append((idx1, idx2))
        for idx2 in other_indices:
            train_negatives.append((idx1, idx2))
for (label, indices) in test_label2idx.items():
    other_indices = set(range(n_test)) - set(indices)
    for (i, idx1) in enumerate(indices):
        for idx2 in indices[i + 1:]:
            test_positives.append((idx1, idx2))
        for idx2 in other_indices:
            test_negatives.append((idx1, idx2))
batch_size = 64

def train_generator():
    if False:
        print('Hello World!')
    n_batches = int(np.ceil(len(train_positives) / batch_size))
    while True:
        np.random.shuffle(train_positives)
        n_samples = batch_size * 2
        shape = [n_samples] + list(img.shape)
        x_batch_1 = np.zeros(shape)
        x_batch_2 = np.zeros(shape)
        y_batch = np.zeros(n_samples)
        for i in range(n_batches):
            pos_batch_indices = train_positives[i * batch_size:(i + 1) * batch_size]
            j = 0
            for (idx1, idx2) in pos_batch_indices:
                x_batch_1[j] = train_images[idx1]
                x_batch_2[j] = train_images[idx2]
                y_batch[j] = 1
                j += 1
            neg_indices = np.random.choice(len(train_negatives), size=len(pos_batch_indices), replace=False)
            for neg in neg_indices:
                (idx1, idx2) = train_negatives[neg]
                x_batch_1[j] = train_images[idx1]
                x_batch_2[j] = train_images[idx2]
                y_batch[j] = 0
                j += 1
            x1 = x_batch_1[:j]
            x2 = x_batch_2[:j]
            y = y_batch[:j]
            yield ([x1, x2], y)

def test_generator():
    if False:
        return 10
    n_batches = int(np.ceil(len(test_positives) / batch_size))
    while True:
        n_samples = batch_size * 2
        shape = [n_samples] + list(img.shape)
        x_batch_1 = np.zeros(shape)
        x_batch_2 = np.zeros(shape)
        y_batch = np.zeros(n_samples)
        for i in range(n_batches):
            pos_batch_indices = test_positives[i * batch_size:(i + 1) * batch_size]
            j = 0
            for (idx1, idx2) in pos_batch_indices:
                x_batch_1[j] = test_images[idx1]
                x_batch_2[j] = test_images[idx2]
                y_batch[j] = 1
                j += 1
            neg_indices = np.random.choice(len(test_negatives), size=len(pos_batch_indices), replace=False)
            for neg in neg_indices:
                (idx1, idx2) = test_negatives[neg]
                x_batch_1[j] = test_images[idx1]
                x_batch_2[j] = test_images[idx2]
                y_batch[j] = 0
                j += 1
            x1 = x_batch_1[:j]
            x2 = x_batch_2[:j]
            y = y_batch[:j]
            yield ([x1, x2], y)
i = Input(shape=img.shape)
x = Conv2D(filters=32, kernel_size=(3, 3))(i)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D()(x)
x = Conv2D(filters=64, kernel_size=(3, 3))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D()(x)
x = Flatten()(x)
x = Dense(units=128, activation='relu')(x)
x = Dense(units=50)(x)
cnn = Model(inputs=i, outputs=x)
img_placeholder1 = Input(shape=img.shape)
img_placeholder2 = Input(shape=img.shape)
feat1 = cnn(img_placeholder1)
feat2 = cnn(img_placeholder2)

def euclidean_distance(features):
    if False:
        print('Hello World!')
    (x, y) = features
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))
dist_layer = Lambda(euclidean_distance)([feat1, feat2])
model = Model(inputs=[img_placeholder1, img_placeholder2], outputs=dist_layer)

def contrastive_loss(y_true, y_pred):
    if False:
        for i in range(10):
            print('nop')
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))
model.compile(loss=contrastive_loss, optimizer='adam')

def get_train_accuracy(threshold=0.85):
    if False:
        while True:
            i = 10
    positive_distances = []
    negative_distances = []
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    batch_size = 64
    x_batch_1 = np.zeros([batch_size] + list(img.shape))
    x_batch_2 = np.zeros([batch_size] + list(img.shape))
    n_batches = int(np.ceil(len(train_positives) / batch_size))
    for i in range(n_batches):
        print(f'pos batch: {i + 1}/{n_batches}')
        pos_batch_indices = train_positives[i * batch_size:(i + 1) * batch_size]
        j = 0
        for (idx1, idx2) in pos_batch_indices:
            x_batch_1[j] = train_images[idx1]
            x_batch_2[j] = train_images[idx2]
            j += 1
        x1 = x_batch_1[:j]
        x2 = x_batch_2[:j]
        distances = model.predict([x1, x2]).flatten()
        positive_distances += distances.tolist()
        tp += (distances < threshold).sum()
        fn += (distances > threshold).sum()
    n_batches = int(np.ceil(len(train_negatives) / batch_size))
    for i in range(n_batches):
        print(f'neg batch: {i + 1}/{n_batches}')
        neg_batch_indices = train_negatives[i * batch_size:(i + 1) * batch_size]
        j = 0
        for (idx1, idx2) in neg_batch_indices:
            x_batch_1[j] = train_images[idx1]
            x_batch_2[j] = train_images[idx2]
            j += 1
        x1 = x_batch_1[:j]
        x2 = x_batch_2[:j]
        distances = model.predict([x1, x2]).flatten()
        negative_distances += distances.tolist()
        fp += (distances < threshold).sum()
        tn += (distances > threshold).sum()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    print(f'sensitivity (tpr): {tpr}, specificity (tnr): {tnr}')
    plt.hist(negative_distances, bins=20, density=True, label='negative_distances')
    plt.hist(positive_distances, bins=20, density=True, label='positive_distances')
    plt.legend()
    plt.show()

def get_test_accuracy(threshold=0.85):
    if False:
        while True:
            i = 10
    positive_distances = []
    negative_distances = []
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    batch_size = 64
    x_batch_1 = np.zeros([batch_size] + list(img.shape))
    x_batch_2 = np.zeros([batch_size] + list(img.shape))
    n_batches = int(np.ceil(len(test_positives) / batch_size))
    for i in range(n_batches):
        print(f'pos batch: {i + 1}/{n_batches}')
        pos_batch_indices = test_positives[i * batch_size:(i + 1) * batch_size]
        j = 0
        for (idx1, idx2) in pos_batch_indices:
            x_batch_1[j] = test_images[idx1]
            x_batch_2[j] = test_images[idx2]
            j += 1
        x1 = x_batch_1[:j]
        x2 = x_batch_2[:j]
        distances = model.predict([x1, x2]).flatten()
        positive_distances += distances.tolist()
        tp += (distances < threshold).sum()
        fn += (distances > threshold).sum()
    n_batches = int(np.ceil(len(test_negatives) / batch_size))
    for i in range(n_batches):
        print(f'neg batch: {i + 1}/{n_batches}')
        neg_batch_indices = test_negatives[i * batch_size:(i + 1) * batch_size]
        j = 0
        for (idx1, idx2) in neg_batch_indices:
            x_batch_1[j] = test_images[idx1]
            x_batch_2[j] = test_images[idx2]
            j += 1
        x1 = x_batch_1[:j]
        x2 = x_batch_2[:j]
        distances = model.predict([x1, x2]).flatten()
        negative_distances += distances.tolist()
        fp += (distances < threshold).sum()
        tn += (distances > threshold).sum()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    print(f'sensitivity (tpr): {tpr}, specificity (tnr): {tnr}')
    plt.hist(negative_distances, bins=20, density=True, label='negative_distances')
    plt.hist(positive_distances, bins=20, density=True, label='positive_distances')
    plt.legend()
    plt.show()
train_steps = int(np.ceil(len(train_positives) * 2 / batch_size))
valid_steps = int(np.ceil(len(test_positives) * 2 / batch_size))
r = model.fit(train_generator(), steps_per_epoch=train_steps, epochs=20, validation_data=test_generator(), validation_steps=valid_steps)
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
get_train_accuracy()
get_test_accuracy()