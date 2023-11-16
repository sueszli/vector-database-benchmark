from AlgorithmImports import *
import torch
import torch.nn.functional as F

class PytorchNeuralNetworkAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 8)
        self.SetCash(100000)
        spy = self.AddEquity('SPY', Resolution.Minute)
        self.symbols = [spy.Symbol]
        self.lookback = 30
        self.Schedule.On(self.DateRules.EveryDay('SPY'), self.TimeRules.AfterMarketOpen('SPY', 28), self.NetTrain)
        self.Schedule.On(self.DateRules.EveryDay('SPY'), self.TimeRules.AfterMarketOpen('SPY', 30), self.Trade)

    def NetTrain(self):
        if False:
            for i in range(10):
                print('nop')
        history = self.History(self.symbols, self.lookback + 1, Resolution.Daily)
        self.prices_x = {}
        self.prices_y = {}
        self.sell_prices = {}
        self.buy_prices = {}
        for symbol in self.symbols:
            if not history.empty:
                self.prices_x[symbol] = list(history.loc[symbol.Value]['open'])[:-1]
                self.prices_y[symbol] = list(history.loc[symbol.Value]['open'])[1:]
        for symbol in self.symbols:
            if symbol in self.prices_x:
                net = Net(n_feature=1, n_hidden=10, n_output=1)
                optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
                loss_func = torch.nn.MSELoss()
                for t in range(200):
                    x = torch.from_numpy(np.array(self.prices_x[symbol])).float()
                    y = torch.from_numpy(np.array(self.prices_y[symbol])).float()
                    x = x.unsqueeze(1)
                    y = y.unsqueeze(1)
                    prediction = net(x)
                    loss = loss_func(prediction, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            self.buy_prices[symbol] = net(y)[-1] + np.std(y.data.numpy())
            self.sell_prices[symbol] = net(y)[-1] - np.std(y.data.numpy())

    def Trade(self):
        if False:
            return 10
        ' \n        Enter or exit positions based on relationship of the open price of the current bar and the prices defined by the machine learning model.\n        Liquidate if the open price is below the sell price and buy if the open price is above the buy price \n        '
        for holding in self.Portfolio.Values:
            if self.CurrentSlice[holding.Symbol].Open < self.sell_prices[holding.Symbol] and holding.Invested:
                self.Liquidate(holding.Symbol)
            if self.CurrentSlice[holding.Symbol].Open > self.buy_prices[holding.Symbol] and (not holding.Invested):
                self.SetHoldings(holding.Symbol, 1 / len(self.symbols))

class Net(torch.nn.Module):

    def __init__(self, n_feature, n_hidden, n_output):
        if False:
            while True:
                i = 10
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        if False:
            return 10
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x