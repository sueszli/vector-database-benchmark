from AlgorithmImports import *
from keras.models import *
from tensorflow import keras
from keras.layers import Dense, Activation
from keras.optimizers import SGD

class KerasNeuralNetworkAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        self.SetStartDate(2019, 1, 1)
        self.SetEndDate(2020, 4, 1)
        self.SetCash(100000)
        self.modelBySymbol = {}
        for ticker in ['SPY', 'QQQ', 'TLT']:
            symbol = self.AddEquity(ticker).Symbol
            for kvp in self.ObjectStore:
                key = f'{symbol}_model'
                if not key == kvp.Key or kvp.Value is None:
                    continue
                filePath = self.ObjectStore.GetFilePath(kvp.Key)
                self.modelBySymbol[symbol] = keras.models.load_model(filePath)
                self.Debug(f'Model for {symbol} sucessfully retrieved. File {filePath}. Size {kvp.Value.Length}. Weights {self.modelBySymbol[symbol].get_weights()}')
        self.lookback = 30
        self.Train(self.DateRules.Every(DayOfWeek.Monday), self.TimeRules.AfterMarketOpen('SPY'), self.NeuralNetworkTraining)
        self.Schedule.On(self.DateRules.EveryDay('SPY'), self.TimeRules.AfterMarketOpen('SPY', 30), self.Trade)

    def OnEndOfAlgorithm(self):
        if False:
            print('Hello World!')
        ' Save the data and the mode using the ObjectStore '
        for (symbol, model) in self.modelBySymbol.items():
            key = f'{symbol}_model'
            file = self.ObjectStore.GetFilePath(key)
            model.save(file)
            self.ObjectStore.Save(key)
            self.Debug(f'Model for {symbol} sucessfully saved in the ObjectStore')

    def NeuralNetworkTraining(self):
        if False:
            while True:
                i = 10
        'Train the Neural Network and save the model in the ObjectStore'
        symbols = self.Securities.keys()
        history = self.History(symbols, self.lookback + 1, Resolution.Daily)
        history = history.open.unstack(0)
        for symbol in symbols:
            if symbol not in history:
                continue
            predictor = history[symbol][:-1]
            predictand = history[symbol][1:]
            model = Sequential()
            model.add(Dense(10, input_dim=1))
            model.add(Activation('relu'))
            model.add(Dense(1))
            sgd = SGD(lr=0.01)
            model.compile(loss='mse', optimizer=sgd)
            for step in range(200):
                cost = model.train_on_batch(predictor, predictand)
            self.modelBySymbol[symbol] = model

    def Trade(self):
        if False:
            while True:
                i = 10
        '\n        Predict the price using the trained model and out-of-sample data\n        Enter or exit positions based on relationship of the open price of the current bar and the prices defined by the machine learning model.\n        Liquidate if the open price is below the sell price and buy if the open price is above the buy price\n        '
        target = 1 / len(self.Securities)
        for (symbol, model) in self.modelBySymbol.items():
            history = self.History(symbol, self.lookback, Resolution.Daily)
            history = history.open.unstack(0)[symbol]
            prediction = model.predict(history)[0][-1]
            historyStd = np.std(history)
            holding = self.Portfolio[symbol]
            openPrice = self.CurrentSlice[symbol].Open
            if holding.Invested:
                if openPrice < prediction - historyStd:
                    self.Liquidate(symbol)
            elif openPrice > prediction + historyStd:
                self.SetHoldings(symbol, target)