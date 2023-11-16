from jesse.enums import timeframes

def timeframe_to_interval(timeframe: str) -> str:
    if False:
        return 10
    if timeframe == timeframes.MINUTE_1:
        return '60'
    else:
        raise NotImplemented('Invalid timeframe: {}'.format(timeframe))

def interval_to_timeframe(interval: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    if interval == '60':
        return timeframes.MINUTE_1
    else:
        raise NotImplemented('Invalid interval: {}'.format(interval))