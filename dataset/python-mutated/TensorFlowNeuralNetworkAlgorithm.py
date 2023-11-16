from AlgorithmImports import *
import tensorflow.compat.v1 as tf

class TensorFlowNeuralNetworkAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 8)
        self.SetCash(100000)
        spy = self.AddEquity('SPY', Resolution.Minute)
        self.symbols = [spy.Symbol]
        self.lookback = 30
        self.Schedule.On(self.DateRules.Every(DayOfWeek.Monday), self.TimeRules.AfterMarketOpen('SPY', 28), self.NetTrain)
        self.Schedule.On(self.DateRules.Every(DayOfWeek.Monday), self.TimeRules.AfterMarketOpen('SPY', 30), self.Trade)

    def add_layer(self, inputs, in_size, out_size, activation_function=None):
        if False:
            return 10
        Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

    def NetTrain(self):
        if False:
            return 10
        history = self.History(self.symbols, self.lookback + 1, Resolution.Daily)
        (self.prices_x, self.prices_y) = ({}, {})
        (self.sell_prices, self.buy_prices) = ({}, {})
        for symbol in self.symbols:
            if not history.empty:
                self.prices_x[symbol] = list(history.loc[symbol.Value]['open'][:-1])
                self.prices_y[symbol] = list(history.loc[symbol.Value]['open'][1:])
        for symbol in self.symbols:
            if symbol in self.prices_x:
                x_data = np.array(self.prices_x[symbol]).astype(np.float32).reshape((-1, 1))
                y_data = np.array(self.prices_y[symbol]).astype(np.float32).reshape((-1, 1))
                tf.disable_v2_behavior()
                xs = tf.placeholder(tf.float32, [None, 1])
                ys = tf.placeholder(tf.float32, [None, 1])
                l1 = self.add_layer(xs, 1, 10, activation_function=tf.nn.relu)
                prediction = self.add_layer(l1, 10, 1, activation_function=None)
                loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
                train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
                sess = tf.Session()
                init = tf.global_variables_initializer()
                sess.run(init)
                for i in range(200):
                    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
            y_pred_final = sess.run(prediction, feed_dict={xs: y_data})[0][-1]
            self.sell_prices[symbol] = y_pred_final - np.std(y_data)
            self.buy_prices[symbol] = y_pred_final + np.std(y_data)

    def Trade(self):
        if False:
            i = 10
            return i + 15
        ' \n        Enter or exit positions based on relationship of the open price of the current bar and the prices defined by the machine learning model.\n        Liquidate if the open price is below the sell price and buy if the open price is above the buy price \n        '
        for holding in self.Portfolio.Values:
            if self.CurrentSlice[holding.Symbol].Open < self.sell_prices[holding.Symbol] and holding.Invested:
                self.Liquidate(holding.Symbol)
            if self.CurrentSlice[holding.Symbol].Open > self.buy_prices[holding.Symbol] and (not holding.Invested):
                self.SetHoldings(holding.Symbol, 1 / len(self.symbols))