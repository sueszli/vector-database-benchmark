import tensorflow as tf
import numpy as np

def min_max_scaler(data):
    if False:
        print('Hello World!')
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-07)
xy = np.array([[828.659973, 833.450012, 908100, 828.349976, 831.659973], [823.02002, 828.070007, 1828100, 821.655029, 828.070007], [819.929993, 824.400024, 1438100, 818.97998, 824.159973], [816, 820.958984, 1008100, 815.48999, 819.23999], [819.359985, 823, 1188100, 818.469971, 818.97998], [819, 823, 1198100, 816, 820.450012], [811.700012, 815.25, 1098100, 809.780029, 813.669983], [809.51001, 816.659973, 1398100, 804.539978, 809.559998]])
xy = min_max_scaler(xy)
print(xy)
'\n[[0.99999999 0.99999999 0.         1.         1.        ]\n [0.70548491 0.70439552 1.         0.71881782 0.83755791]\n [0.54412549 0.50274824 0.57608696 0.606468   0.6606331 ]\n [0.33890353 0.31368023 0.10869565 0.45989134 0.43800918]\n [0.51436    0.42582389 0.30434783 0.58504805 0.42624401]\n [0.49556179 0.42582389 0.31521739 0.48131134 0.49276137]\n [0.11436064 0.         0.20652174 0.22007776 0.18597238]\n [0.         0.07747099 0.5326087  0.         0.        ]]\n'
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]
tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=4))
tf.model.add(tf.keras.layers.Activation('linear'))
tf.model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(lr=1e-05))
tf.model.summary()
history = tf.model.fit(x_data, y_data, epochs=1000)
predictions = tf.model.predict(x_data)
score = tf.model.evaluate(x_data, y_data)
print('Prediction: \n', predictions)
print('Cost: ', score)