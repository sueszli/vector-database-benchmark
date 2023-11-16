#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import matplotlib as mpl
import matplotlib.pyplot as plt
#import seaborn as sns
import h5py 
import os 
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from sklearn.metrics import f1_score


# In[6]:
x_train=pd.read_csv('Data/X_train_100.csv',header=None)
x_train = x_train.to_numpy()
x_test=pd.read_csv('Data/X_test_100.csv',header=None)
y_test1=pd.read_csv('Data/Y_test_100.csv',header=None)
x_test = x_test.to_numpy()



# In[7]:

y_test1.columns=['y']
print(x_test.shape)
print(y_test1['y'].shape)


# In[8]:


x_train = x_train.reshape((8000, 3000, 1)).astype("float32")
x_test = x_test.reshape(1000, 3000,1).astype("float32")
y_test = keras.utils.to_categorical(y_test1)


# In[9]:

x_test = (x_test-x_train.mean())/x_train.std();


print(x_test.shape)
print(y_test.shape)

# In[12]:

model = tf.keras.models.load_model('Model1.h5')
model.summary()

# In[15]:

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# In[17]:

predict_x=model.predict(x_test) 
np.savetxt('Raw_prediction_cnn.txt',predict_x)
classes_x=np.argmax(predict_x,axis=1)
print(classes_x)
np.savetxt('Prediction_test.txt',classes_x)
con_mat = tf.math.confusion_matrix(labels=y_test1, predictions=classes_x).numpy()
classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99]
con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
con_mat_df = pd.DataFrame(con_mat_norm,index = classes, columns = classes)
con_mat_df.to_csv('confusion_matrix_test.csv')

print("F1-Score:",f1_score(y_test1, classes_x, average='macro'))



