"""
Regression tasks estimate a numeric variable, such as the price of a house or
voter turnout.

This example is adapted from a
[notebook](https://gist.github.com/mapmeld/98d1e9839f2d1f9c4ee197953661ed07)
which estimates a person's age from their image, trained on the
[IMDB-WIKI](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) photographs
of famous
people.

First, prepare your image data in a numpy.ndarray or tensorflow.Dataset format.
Each image must have the same shape, meaning each has the same width, height,
and color channels as other images in the set.
"""
from datetime import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
import tensorflow as tf
from google.colab import drive
from PIL import Image
from scipy.io import loadmat
import autokeras as ak
'\n### Connect your Google Drive for Data\n'
drive.mount('/content/drive')
'\n### Install AutoKeras and TensorFlow\n\nDownload the master branch to your Google Drive for this tutorial. In general,\nyou can use *pip install autokeras* .\n'
'shell\n!pip install  -v "/content/drive/My Drive/AutoKeras-dev/autokeras-master.zip"\n!pip uninstall keras-tuner\n!pip install\ngit+git://github.com/keras-team/keras-tuner.git@d2d69cba21a0b482a85ce2a38893e2322e139c01\n'
'shell\n!pip install tensorflow==2.2.0\n'
'\n###**Import IMDB Celeb images and metadata**\n'
'shell\n!mkdir "./drive/My Drive/mlin/celebs"\n'
'shell\n! wget -O "./drive/My Drive/mlin/celebs/imdb_0.tar"\nhttps://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_0.tar\n'
'shell\n! cd "./drive/My Drive/mlin/celebs" && tar -xf imdb_0.tar\n! rm "./drive/My Drive/mlin/celebs/imdb_0.tar"\n'
"\nUncomment and run the below cell if you need to re-run the cells again and\nabove don't need to install everything from the beginning.\n"
'shell\n! ls "./drive/My Drive/mlin/celebs/imdb/"\n'
'shell\n! wget https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_meta.tar\n! tar -xf imdb_meta.tar\n! rm imdb_meta.tar\n'
'\n###**Converting from MATLAB date to actual Date-of-Birth**\n'

def datenum_to_datetime(datenum):
    if False:
        return 10
    '\n    Convert Matlab datenum into Python datetime.\n    '
    days = datenum % 1
    hours = days % 1 * 24
    minutes = hours % 1 * 60
    seconds = minutes % 1 * 60
    try:
        return datetime.fromordinal(int(datenum)) + timedelta(days=int(days)) + timedelta(hours=int(hours)) + timedelta(minutes=int(minutes)) + timedelta(seconds=round(seconds)) - timedelta(days=366)
    except Exception:
        return datenum_to_datetime(700000)
print(datenum_to_datetime(734963))
'\n### **Opening MatLab file to Pandas DataFrame**\n'
x = loadmat('imdb/imdb.mat')
mdata = x['imdb']
mdtype = mdata.dtype
ndata = {n: mdata[n][0, 0] for n in mdtype.names}
columns = [n for (n, v) in ndata.items()]
rows = []
for col in range(0, 10):
    values = list(ndata.items())[col]
    for (num, val) in enumerate(values[1][0], start=0):
        if col == 0:
            rows.append([])
        if num > 0:
            if columns[col] == 'dob':
                rows[num].append(datenum_to_datetime(int(val)))
            elif columns[col] == 'photo_taken':
                rows[num].append(datetime(year=int(val), month=6, day=30))
            else:
                rows[num].append(val)
dt = map(lambda row: np.array(row), np.array(rows[1:]))
df = pd.DataFrame(data=dt, index=range(0, len(rows) - 1), columns=columns)
print(df.head())
print(columns)
print(df['full_path'])
'\n### **Calculating age at time photo was taken**\n'
df['age'] = (df['photo_taken'] - df['dob']).astype('int') / 3.1558102e+16
print(df['age'])
'\n### **Creating dataset**\n\n\n* We sample 200 of the images which were included in this first download.\n* Images are resized to 128x128 to standardize shape and conserve memory\n* RGB images are converted to grayscale to standardize shape\n* Ages are converted to ints\n\n\n'

def df2numpy(train_set):
    if False:
        print('Hello World!')
    images = []
    for img_path in train_set['full_path']:
        img = Image.open('./drive/My Drive/mlin/celebs/imdb/' + img_path[0]).resize((128, 128)).convert('L')
        images.append(np.asarray(img, dtype='int32'))
    image_inputs = np.array(images)
    ages = train_set['age'].astype('int').to_numpy()
    return (image_inputs, ages)
train_set = df[df['full_path'] < '02'].sample(200)
(train_imgs, train_ages) = df2numpy(train_set)
test_set = df[df['full_path'] < '02'].sample(100)
(test_imgs, test_ages) = df2numpy(test_set)
'\n### **Training using AutoKeras**\n'
reg = ak.ImageRegressor(max_trials=15)
reg.fit(train_imgs, train_ages)
print(reg.evaluate(train_imgs, train_ages))
'\n### **Validation Data**\n\nBy default, AutoKeras use the last 20% of training data as validation data. As\nshown in the example below, you can use validation_split to specify the\npercentage.\n'
reg.fit(train_imgs, train_ages, validation_split=0.15, epochs=3)
'\nYou can also use your own validation set instead of splitting it from the\ntraining data with validation_data.\n'
split = 460000
x_val = train_imgs[split:]
y_val = train_ages[split:]
x_train = train_imgs[:split]
y_train = train_ages[:split]
reg.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=3)
'\n### **Customized Search Space**\n\nFor advanced users, you may customize your search space by using AutoModel\ninstead of ImageRegressor. You can configure the ImageBlock for some high-level\nconfigurations, e.g., block_type for the type of neural network to search,\nnormalize for whether to do data normalization, augment for whether to do data\naugmentation. You can also choose not to specify these arguments, which would\nleave the different choices to be tuned automatically. See the following\nexample for detail.\n'
input_node = ak.ImageInput()
output_node = ak.ImageBlock(block_type='resnet', normalize=True, augment=False)(input_node)
output_node = ak.RegressionHead()(output_node)
reg = ak.AutoModel(inputs=input_node, outputs=output_node, max_trials=10)
reg.fit(x_train, y_train, epochs=3)
'\nThe usage of AutoModel is similar to the functional API of Keras. Basically, you\nare building a graph, whose edges are blocks and the nodes are intermediate\noutputs of blocks. To add an edge from input_node to output_node with\noutput_node = ak.some_block(input_node).  You can even also use more fine\ngrained blocks to customize the search space even further. See the following\nexample.\n'
input_node = ak.ImageInput()
output_node = ak.Normalization()(input_node)
output_node = ak.ImageAugmentation(translation_factor=0.3)(output_node)
output_node = ak.ResNetBlock(version='v2')(output_node)
output_node = ak.RegressionHead()(output_node)
clf = ak.AutoModel(inputs=input_node, outputs=output_node, max_trials=10)
clf.fit(x_train, y_train, epochs=3)
'\n### **Data Format**\n'
'\nThe AutoKeras ImageClassifier is quite flexible for the data format.\n\nFor the image, it accepts data formats both with and without the channel\ndimension. The images in the IMDB-Wiki dataset do not have a channel dimension.\nEach image is a matrix with shape (128, 128). AutoKeras also accepts images\nwith a channel dimension at last, e.g., (32, 32, 3), (28, 28, 1).\n\nFor the classification labels, AutoKeras accepts both plain labels, i.e.\nstrings or integers, and one-hot encoded labels, i.e. vectors of 0s and 1s.\n\nSo if you prepare your data in the following way, the ImageClassifier should\nstill work.\n'
train_imgs = train_imgs.reshape(train_imgs.shape + (1,))
test_imgs = test_imgs.reshape(test_imgs.shape + (1,))
print(train_imgs.shape)
print(test_imgs.shape)
print(train_ages[:3])
'\nWe also support using tf.data.Dataset format for the training data. In this\ncase, the images would have to be 3-dimentional. The labels have to be one-hot\nencoded for multi-class classification to be wrapped into tensorflow Dataset.\n'
train_set = tf.data.Dataset.from_tensor_slices(((train_imgs,), (train_ages,)))
test_set = tf.data.Dataset.from_tensor_slices(((test_imgs,), (test_ages,)))
reg = ak.ImageRegressor(max_trials=15)
reg.fit(train_set)
predicted_y = clf.predict(test_set)
print(clf.evaluate(test_set))
'\n## References\n\n[Main Reference\nNotebook](https://gist.github.com/mapmeld/98d1e9839f2d1f9c4ee197953661ed07),\n[Dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/),\n[ImageRegressor](/image_regressor),\n[ResNetBlock](/block/#resnetblock-class),\n[ImageInput](/node/#imageinput-class),\n[AutoModel](/auto_model/#automodel-class),\n[ImageBlock](/block/#imageblock-class),\n[Normalization](/preprocessor/#normalization-class),\n[ImageAugmentation](/preprocessor/#image-augmentation-class),\n[RegressionHead](/head/#regressionhead-class).\n\n'