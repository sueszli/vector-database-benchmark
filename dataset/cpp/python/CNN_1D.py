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
import h5py
import os
from sklearn.metrics import f1_score
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# In[6]:


x_train=pd.read_csv('Data/X_train_100.csv',header=None)
y_train=pd.read_csv('Data/Y_train_100.csv',header=None)
x_train = x_train.to_numpy()
x_test=pd.read_csv('Data/X_val_100.csv',header=None)
y_test1=pd.read_csv('Data/Y_val_100.csv',header=None)
x_test = x_test.to_numpy()



# In[7]:
y_train.columns=['y']
y_test1.columns=['y']

print(x_train.shape)
print(x_test.shape)
print(y_train['y'].shape)
print(y_test1['y'].shape)


# In[8]:


x_train = x_train.reshape((8000, 3000, 1)).astype("float32")
x_test = x_test.reshape(1000, 3000,1).astype("float32")
# Categorical (one hot) encoding of the labels
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test1)


# In[9]:

mean = x_train.mean()
std = x_train.std()
x_train = (x_train-mean)/std
x_test =  (x_test-mean)/std


# In[10]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[11]:


model = models.Sequential()
model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(3000,1)))
model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(tf.keras.layers.MaxPooling1D(pool_size=3))
#model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(tf.keras.layers.MaxPooling1D(pool_size=3))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(128, activation='relu'))
#model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(100, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[12]:


model.summary()


# In[13]:


history = model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test))


# In[14]:


np.save('History1.npy',history.history)
model.save('Model1.h5')

# In[15]:


score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


# In[17]:


predict_x=model.predict(x_test) 
#np.savetxt('Raw_prediction.txt',predict_x)
classes_x=np.argmax(predict_x,axis=1)
print(classes_x)
#np.savetxt('Prediction2.txt',classes_x)
con_mat = tf.math.confusion_matrix(labels=y_test1, predictions=classes_x).numpy()
classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99]
con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
con_mat_df = pd.DataFrame(con_mat_norm,index = classes, columns = classes)
con_mat_df.to_csv('Confusion_matrix_val.csv')

print("F1-Score:",f1_score(y_test1, classes_x, average='macro'))



