from AlgorithmImports import *
import nltk

class NLTKSentimentTradingAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            print('Hello World!')
        self.SetStartDate(2018, 1, 1)
        self.SetEndDate(2019, 1, 1)
        self.SetCash(100000)
        spy = self.AddEquity('SPY', Resolution.Minute)
        self.text = self.get_text()
        self.symbols = [spy.Symbol]
        nltk.download('punkt')
        self.Schedule.On(self.DateRules.EveryDay('SPY'), self.TimeRules.AfterMarketOpen('SPY', 30), self.Trade)

    def Trade(self):
        if False:
            while True:
                i = 10
        current_time = f'{self.Time.year}-{self.Time.month}-{self.Time.day}'
        current_text = self.text.loc[current_time][0]
        words = nltk.word_tokenize(current_text)
        positive_word = 'Up'
        negative_word = 'Down'
        for holding in self.Portfolio.Values:
            if negative_word in words and holding.Invested:
                self.Liquidate(holding.Symbol)
            if positive_word in words and (not holding.Invested):
                self.SetHoldings(holding.Symbol, 1 / len(self.symbols))

    def get_text(self):
        if False:
            while True:
                i = 10
        url = 'https://www.dropbox.com/s/7xgvkypg6uxp6xl/EconomicNews.csv?dl=1'
        data = self.Download(url).split('\n')
        headline = [x.split(',')[1] for x in data][1:]
        date = [x.split(',')[0] for x in data][1:]
        df = pd.DataFrame(headline, index=date, columns=['headline'])
        return df