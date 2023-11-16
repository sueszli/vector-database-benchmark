import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

def min_max_scaler(data):
    if False:
        while True:
            i = 10
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-07)
xy = np.array([[828.659973, 833.450012, 908100, 828.349976, 831.659973], [823.02002, 828.070007, 1828100, 821.655029, 828.070007], [819.929993, 824.400024, 1438100, 818.97998, 824.159973], [816, 820.958984, 1008100, 815.48999, 819.23999], [819.359985, 823, 1188100, 818.469971, 818.97998], [819, 823, 1198100, 816, 820.450012], [811.700012, 815.25, 1098100, 809.780029, 813.669983], [809.51001, 816.659973, 1398100, 804.539978, 809.559998]])
xy = min_max_scaler(xy)
print(xy)
'\n[[0.99999999 0.99999999 0.         1.         1.        ]\n [0.70548491 0.70439552 1.         0.71881782 0.83755791]\n [0.54412549 0.50274824 0.57608696 0.606468   0.6606331 ]\n [0.33890353 0.31368023 0.10869565 0.45989134 0.43800918]\n [0.51436    0.42582389 0.30434783 0.58504805 0.42624401]\n [0.49556179 0.42582389 0.31521739 0.48131134 0.49276137]\n [0.11436064 0.         0.20652174 0.22007776 0.18597238]\n [0.         0.07747099 0.5326087  0.         0.        ]]\n'
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]
X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([4, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))
train = tf.train.GradientDescentOptimizer(learning_rate=1e-05).minimize(cost)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(101):
        (_, cost_val, hy_val) = sess.run([train, cost, hypothesis], feed_dict={X: x_data, Y: y_data})
        print(step, 'Cost: ', cost_val, '\nPrediction:\n', hy_val)
'\n0 Cost: 0.15230925 \nPrediction:\n [[ 1.6346191 ]\n [ 0.06613699]\n [ 0.3500818 ]\n [ 0.6707252 ]\n [ 0.61130744]\n [ 0.61464405]\n [ 0.23171967]\n [-0.1372836 ]]\n1 Cost: 0.15230872 \nPrediction:\n [[ 1.634618  ]\n [ 0.06613836]\n [ 0.35008252]\n [ 0.670725  ]\n [ 0.6113076 ]\n [ 0.6146443 ]\n [ 0.23172   ]\n [-0.13728246]]\n...\n99 Cost: 0.1522546 \nPrediction:\n [[ 1.6345041 ]\n [ 0.06627947]\n [ 0.35014683]\n [ 0.670706  ]\n [ 0.6113161 ]\n [ 0.61466044]\n [ 0.23175153]\n [-0.13716647]]\n100 Cost: 0.15225402 \nPrediction:\n [[ 1.6345029 ]\n [ 0.06628093]\n [ 0.35014752]\n [ 0.67070574]\n [ 0.61131614]\n [ 0.6146606 ]\n [ 0.23175186]\n [-0.13716528]]\n'