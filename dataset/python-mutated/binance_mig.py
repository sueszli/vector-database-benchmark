import logging
from packaging import version
from sqlalchemy import select
from freqtrade.constants import DOCS_LINK, Config
from freqtrade.enums import TradingMode
from freqtrade.exceptions import OperationalException
from freqtrade.persistence.pairlock import PairLock
from freqtrade.persistence.trade_model import Trade
logger = logging.getLogger(__name__)

def migrate_binance_futures_names(config: Config):
    if False:
        i = 10
        return i + 15
    if not (config.get('trading_mode', TradingMode.SPOT) == TradingMode.FUTURES and config['exchange']['name'] == 'binance'):
        return
    import ccxt
    if version.parse('2.6.26') > version.parse(ccxt.__version__):
        raise OperationalException(f'Please follow the update instructions in the docs ({DOCS_LINK}/updating/) to install a compatible ccxt version.')
    _migrate_binance_futures_db(config)
    migrate_binance_futures_data(config)

def _migrate_binance_futures_db(config: Config):
    if False:
        print('Hello World!')
    logger.warning('Migrating binance futures pairs in database.')
    trades = Trade.get_trades([Trade.exchange == 'binance', Trade.trading_mode == 'FUTURES']).all()
    for trade in trades:
        if ':' in trade.pair:
            continue
        new_pair = f'{trade.pair}:{trade.stake_currency}'
        trade.pair = new_pair
        for order in trade.orders:
            order.ft_pair = new_pair
    Trade.commit()
    pls = PairLock.session.scalars(select(PairLock).filter(PairLock.pair.notlike('%:%'))).all()
    for pl in pls:
        pl.pair = f"{pl.pair}:{config['stake_currency']}"
    Trade.commit()
    logger.warning('Done migrating binance futures pairs in database.')

def migrate_binance_futures_data(config: Config):
    if False:
        print('Hello World!')
    if not (config.get('trading_mode', TradingMode.SPOT) == TradingMode.FUTURES and config['exchange']['name'] == 'binance'):
        return
    from freqtrade.data.history.idatahandler import get_datahandler
    dhc = get_datahandler(config['datadir'], config['dataformat_ohlcv'])
    paircombs = dhc.ohlcv_get_available_data(config['datadir'], config.get('trading_mode', TradingMode.SPOT))
    for (pair, timeframe, candle_type) in paircombs:
        if ':' in pair:
            continue
        new_pair = f"{pair}:{config['stake_currency']}"
        dhc.rename_futures_data(pair, new_pair, timeframe, candle_type)