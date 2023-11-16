from AlgorithmImports import *

class IndustryStandardSecurityIdentifiersRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            print('Hello World!')
        self.SetStartDate(2014, 6, 5)
        self.SetEndDate(2014, 6, 5)
        equity = self.AddEquity('AAPL').Symbol
        cusip = equity.CUSIP
        compositeFigi = equity.CompositeFIGI
        sedol = equity.SEDOL
        isin = equity.ISIN
        cik = equity.CIK
        self.CheckSymbolRepresentation(cusip, 'CUSIP')
        self.CheckSymbolRepresentation(compositeFigi, 'Composite FIGI')
        self.CheckSymbolRepresentation(sedol, 'SEDOL')
        self.CheckSymbolRepresentation(isin, 'ISIN')
        self.CheckSymbolRepresentation(f'{cik}', 'CIK')
        self.CheckAPIsSymbolRepresentations(cusip, self.CUSIP(equity), 'CUSIP')
        self.CheckAPIsSymbolRepresentations(compositeFigi, self.CompositeFIGI(equity), 'Composite FIGI')
        self.CheckAPIsSymbolRepresentations(sedol, self.SEDOL(equity), 'SEDOL')
        self.CheckAPIsSymbolRepresentations(isin, self.ISIN(equity), 'ISIN')
        self.CheckAPIsSymbolRepresentations(f'{cik}', f'{self.CIK(equity)}', 'CIK')
        self.Log(f'\nAAPL CUSIP: {cusip}\nAAPL Composite FIGI: {compositeFigi}\nAAPL SEDOL: {sedol}\nAAPL ISIN: {isin}\nAAPL CIK: {cik}')

    def CheckSymbolRepresentation(self, symbol: str, standard: str) -> None:
        if False:
            while True:
                i = 10
        if not symbol:
            raise Exception(f'{standard} symbol representation is null or empty')

    def CheckAPIsSymbolRepresentations(self, symbolApiSymbol: str, algorithmApiSymbol: str, standard: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        if symbolApiSymbol != algorithmApiSymbol:
            raise Exception(f'Symbol API {standard} symbol representation ({symbolApiSymbol}) does not match QCAlgorithm API {standard} symbol representation ({algorithmApiSymbol})')