from tqsdk import TqApi, TqAuth
'\n根据输入的ETF期权来查询该期权的交易所规则下的理论卖方保证金，实际情况请以期货公司收取的一手保证金为准\n'

def etf_margin_cal(symbol):
    if False:
        for i in range(10):
            print('nop')
    quote_etf = api.get_quote(symbol)
    if quote_etf.underlying_symbol in ['SSE.510050', 'SSE.510300', 'SZSE.159919']:
        if quote_etf.option_class == 'CALL':
            call_out_value = max(quote_etf.strike_price - quote_etf.underlying_quote.pre_close, 0)
            call_margin = (quote_etf.pre_settlement + max(0.12 * quote_etf.underlying_quote.pre_close - call_out_value, 0.07 * quote_etf.underlying_quote.pre_close)) * quote_etf.volume_multiple
            return round(call_margin, 2)
        elif quote_etf.option_class == 'PUT':
            put_out_value = max(quote_etf.underlying_quote.pre_close - quote_etf.strike_price, 0)
            put_margin = min(quote_etf.pre_settlement + max(0.12 * quote_etf.underlying_quote.pre_close - put_out_value, 0.07 * quote_etf.strike_price), quote_etf.strike_price) * quote_etf.volume_multiple
            return round(put_margin, 2)
    else:
        print('输入的不是ETF期权合约')
        return None
api = TqApi(auth=TqAuth('快期账户', '账户密码'))
symbol = 'SZSE.90000833'
print(etf_margin_cal(symbol))
api.close()