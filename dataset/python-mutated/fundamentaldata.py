from .alphavantage import AlphaVantage as av
from datetime import datetime
search_date = datetime.now().date().strftime('%Y-%m-%d')

class FundamentalData(av):
    """This class implements all the api calls to fundamental data
    """

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        '\n        Inherit AlphaVantage base class with its default arguments.\n        '
        super(FundamentalData, self).__init__(*args, **kwargs)
        self._append_type = False
        if self.output_format.lower() == 'csv':
            raise ValueError('Output format {} is not compatible with the FundamentalData class'.format(self.output_format.lower()))

    @av._output_format
    @av._call_api_on_func
    def get_company_overview(self, symbol):
        if False:
            return 10
        '\n        Returns the company information, financial ratios, \n        and other key metrics for the equity specified. \n        Data is generally refreshed on the same day a company reports its latest \n        earnings and financials.\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n        '
        _FUNCTION_KEY = 'OVERVIEW'
        return (_FUNCTION_KEY, None, None)

    @av._output_format
    @av._call_api_on_func
    def get_income_statement_annual(self, symbol):
        if False:
            while True:
                i = 10
        '\n        Returns the annual and quarterly income statements for the company of interest. \n        Data is generally refreshed on the same day a company reports its latest \n        earnings and financials.\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n        '
        _FUNCTION_KEY = 'INCOME_STATEMENT'
        return (_FUNCTION_KEY, 'annualReports', 'symbol')

    @av._output_format
    @av._call_api_on_func
    def get_income_statement_quarterly(self, symbol):
        if False:
            print('Hello World!')
        '\n        Returns the annual and quarterly income statements for the company of interest. \n        Data is generally refreshed on the same day a company reports its latest \n        earnings and financials.\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n        '
        _FUNCTION_KEY = 'INCOME_STATEMENT'
        return (_FUNCTION_KEY, 'quarterlyReports', 'symbol')

    @av._output_format
    @av._call_api_on_func
    def get_balance_sheet_annual(self, symbol):
        if False:
            while True:
                i = 10
        '\n        Returns the annual and quarterly balance sheets for the company of interest.\n        Data is generally refreshed on the same day a company reports its latest \n        earnings and financials.\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n        '
        _FUNCTION_KEY = 'BALANCE_SHEET'
        return (_FUNCTION_KEY, 'annualReports', 'symbol')

    @av._output_format
    @av._call_api_on_func
    def get_balance_sheet_quarterly(self, symbol):
        if False:
            return 10
        '\n        Returns the annual and quarterly balance sheets for the company of interest.\n        Data is generally refreshed on the same day a company reports its latest \n        earnings and financials.\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n        '
        _FUNCTION_KEY = 'BALANCE_SHEET'
        return (_FUNCTION_KEY, 'quarterlyReports', 'symbol')

    @av._output_format
    @av._call_api_on_func
    def get_cash_flow_annual(self, symbol):
        if False:
            return 10
        '\n        Returns the annual and quarterly cash flows for the company of interest.\n        Data is generally refreshed on the same day a company reports its latest \n        earnings and financials.\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n        '
        _FUNCTION_KEY = 'CASH_FLOW'
        return (_FUNCTION_KEY, 'annualReports', 'symbol')

    @av._output_format
    @av._call_api_on_func
    def get_cash_flow_quarterly(self, symbol):
        if False:
            while True:
                i = 10
        '\n        Returns the annual and quarterly cash flows for the company of interest.\n        Data is generally refreshed on the same day a company reports its latest \n        earnings and financials.\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n        '
        _FUNCTION_KEY = 'CASH_FLOW'
        return (_FUNCTION_KEY, 'quarterlyReports', 'symbol')