from AlgorithmImports import *

class PandasDataFrameFromMultipleTickTypeTickHistoryRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            while True:
                i = 10
        self.SetStartDate(2013, 10, 8)
        self.SetEndDate(2013, 10, 8)
        spy = self.AddEquity('SPY', Resolution.Minute).Symbol
        subscriptions = [x for x in self.SubscriptionManager.Subscriptions if x.Symbol == spy]
        if len(subscriptions) != 2:
            raise Exception(f'Expected 2 subscriptions, but found {len(subscriptions)}')
        history = pd.DataFrame()
        try:
            history = self.History(Tick, spy, timedelta(days=1), Resolution.Tick)
        except Exception as e:
            raise Exception(f'History call failed: {e}')
        if history.shape[0] == 0:
            raise Exception('SPY tick history is empty')
        if not np.array_equal(history.columns.to_numpy(), ['askprice', 'asksize', 'bidprice', 'bidsize', 'exchange', 'lastprice', 'quantity']):
            raise Exception('Unexpected columns in SPY tick history')
        self.Quit()