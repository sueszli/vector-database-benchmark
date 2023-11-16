from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.preprocessing import image as image_utils
import matplotlib.pyplot as plt
MASKS_DIR = '/home/ubuntu/csc420_data/data_road/training/gt_image_road_mask'
IMAGES_DIR = '/home/ubuntu/csc420_data/data_road/training/image_2'
i_shape = (248, 504)
x_shape_tensor = (248, 504, 3)
y_shape_tensor = (248, 504, 1)

def load_images(dir, grayscale=False):
    if False:
        i = 10
        return i + 15
    imgs = []
    for i in image_utils.list_pictures(dir):
        imgs.append(image_utils.img_to_array(image_utils.load_img(i, grayscale=grayscale, target_size=i_shape)))
    return np.array(imgs)

def load_data():
    if False:
        return 10
    x_data = load_images(IMAGES_DIR)
    y_data = load_images(MASKS_DIR, grayscale=True)
    indices = np.random.permutation(x_data.shape[0])
    (training_idx, test_idx) = (indices[:80], indices[80:])
    (x_training, x_valid) = (x_data[training_idx, :], x_data[test_idx, :])
    (y_training, y_valid) = (y_data[training_idx, :], y_data[test_idx, :])
    return ((x_training, y_training), (x_valid, y_valid))
input_img = Input(shape=x_shape_tensor)
x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(input_img)
x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(16, 5, 5, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(16, 5, 5, activation='relu', border_mode='same')(x)
encoded = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(16, 5, 5, activation='relu', border_mode='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(16, 5, 5, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(64, 3, 3, activation='relu')(x)
x = Convolution2D(64, 3, 3, activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
import numpy as np
((x_train, y_train), (x_test, y_test)) = load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = np.reshape(x_train, (len(x_train), x_shape_tensor[0], x_shape_tensor[1], x_shape_tensor[2]))
x_test = np.reshape(x_test, (len(x_test), x_shape_tensor[0], x_shape_tensor[1], x_shape_tensor[2]))
y_train = y_train.astype('float32') / 255.0
y_test = y_test.astype('float32') / 255.0
y_train = np.reshape(y_train, (len(y_train), y_shape_tensor[0], y_shape_tensor[1], y_shape_tensor[2]))
y_test = np.reshape(y_test, (len(y_test), y_shape_tensor[0], y_shape_tensor[1], y_shape_tensor[2]))
from keras.callbacks import TensorBoard
autoencoder.fit(x_train, y_train, nb_epoch=200, batch_size=16, shuffle=True, validation_data=(x_test, y_test), callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
autoencoder.save('autoencodernov25.h5p')
decoded_imgs = autoencoder.predict(x_test)
n = 10
for i in range(n):
    plt.imsave('input_' + str(i) + '.png', x_test[i].reshape(x_shape_tensor[0], x_shape_tensor[1], x_shape_tensor[2]), cmap='gray')
    plt.imsave('output_' + str(i) + '.png', decoded_imgs[i].reshape(y_shape_tensor[0], y_shape_tensor[1]), cmap='gray')