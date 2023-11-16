import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import losses, metrics
from bigdl.nano.tf.keras import InferenceOptimizer
num_classes = 10
input_shape = (32, 32, 3)
((x_train, y_train), (x_test, y_test)) = keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def creat_model():
    if False:
        for i in range(10):
            print('nop')
    model = keras.Sequential()
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model
if __name__ == '__main__':
    model = creat_model()
    model.summary()
    batch_size = 128
    epochs = 20
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    tune_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    q_model = InferenceOptimizer.quantize(model, x=tune_dataset)
    y_test_hat = q_model(x_test)
    loss = float(tf.reduce_mean(losses.categorical_crossentropy(y_test, y_test_hat)))
    categorical_accuracy = metrics.CategoricalAccuracy()
    categorical_accuracy.update_state(y_test, y_test_hat)
    accuracy = categorical_accuracy.result().numpy()
    print('Quantization test loss:', loss)
    print('Quantization test accuracy:', accuracy)