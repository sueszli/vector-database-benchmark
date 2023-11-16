###
#inspired by https://pythonprogramming.net/tensorboard-optimizing-models-deep-learning-python-tensorflow-keras/
#
###
from utils import *
from classification_LDA import test_train_set


def train_test_valid_CNN(data):

    #create
    feature_train,feature_test,label_train,label_test = test_train_set(data)
    feature_train = np.expand_dims(feature_train.values, axis=2)
    feature_test = np.expand_dims(feature_test.values, axis=2)


    #define parameters for model
    dense_layers = [0, 1, 2]
    layer_sizes = [32, 64, 128]
    conv_layers = [1, 2, 3]

    # #create the model for each parameter combination
    # for dense_layer in dense_layers:
    #     for layer_size in layer_sizes:
    #         for conv_layer in conv_layers:
    #             model_name = f"2-Labels-binaryCrossEntropy-softmax-{conv_layer}-conv-{layer_size}-nodes-{dense_layer}-dense-",int(time.time())
    #             print(model_name)
    #             model = Sequential()
    #             model.add(Conv1D(filters=layer_size,kernel_size=3, activation='relu',
    #                              input_shape=(feature_train.shape[1], feature_train.shape[2])))
    #             model.add(MaxPooling1D(pool_size=2))
    #
    #             for l in range(conv_layer - 1):
    #                 model.add(Conv1D(filters=layer_size, kernel_size=3, padding='same', activation='relu'))
    #                 model.add(MaxPooling1D(pool_size=2))
    #
    #             model.add(Flatten())
    #             for _ in range(dense_layer):
    #                 model.add(Dense(layer_size, activation='relu'))
    #
    #             model.add(Dense(1, activation='softmax'))
    #
    #             tensorboard = TensorBoard(log_dir=f"logs/{model_name}")
    #
    #             # Compile the model
    #             model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #
    #             # Train the model
    #             model.fit(feature_train, label_train, epochs=15, batch_size=32, validation_data=(feature_test, label_test), callbacks=[tensorboard])

    model_name = f"2-Labels-binary_crossentropy-sigmoid-{3}-conv-{64}-nodes-{1}-dense-", int(time.time())
    print(model_name)
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu',
                     input_shape=(feature_train.shape[1], feature_train.shape[2])))
    model.add(MaxPooling1D(pool_size=2))

    #for l in range(conv_layer - 1):
    model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())

    model.add(Dense(1, activation='sigmoid'))

    tensorboard = TensorBoard(log_dir=f"logs/{model_name}")

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(feature_train, label_train, epochs=10, batch_size=32, validation_data=(feature_test, label_test),
              callbacks=[tensorboard])