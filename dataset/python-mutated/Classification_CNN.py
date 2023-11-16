from utils import *
from classification_LDA import test_train_set

def train_test_valid_CNN(data):
    if False:
        i = 10
        return i + 15
    (feature_train, feature_test, label_train, label_test) = test_train_set(data)
    feature_train = np.expand_dims(feature_train.values, axis=2)
    feature_test = np.expand_dims(feature_test.values, axis=2)
    dense_layers = [0, 1, 2]
    layer_sizes = [32, 64, 128]
    conv_layers = [1, 2, 3]
    model_name = (f'2-Labels-binary_crossentropy-sigmoid-{3}-conv-{64}-nodes-{1}-dense-', int(time.time()))
    print(model_name)
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(feature_train.shape[1], feature_train.shape[2])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    tensorboard = TensorBoard(log_dir=f'logs/{model_name}')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(feature_train, label_train, epochs=10, batch_size=32, validation_data=(feature_test, label_test), callbacks=[tensorboard])