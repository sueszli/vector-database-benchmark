import shutil
import pytest
from freqtrade.persistence import Trade
from freqtrade.util.binance_mig import migrate_binance_futures_data, migrate_binance_futures_names
from tests.conftest import create_mock_trades_usdt, log_has

def test_binance_mig_data_conversion(default_conf_usdt, tmp_path, testdatadir):
    if False:
        i = 10
        return i + 15
    migrate_binance_futures_data(default_conf_usdt)
    default_conf_usdt['trading_mode'] = 'futures'
    pair_old = 'XRP_USDT'
    pair_unified = 'XRP_USDT_USDT'
    futures_src = testdatadir / 'futures'
    futures_dst = tmp_path / 'futures'
    futures_dst.mkdir()
    files = ['-1h-mark.feather', '-1h-futures.feather', '-8h-funding_rate.feather', '-8h-mark.feather']
    for file in files:
        fn_after = futures_dst / f'{pair_old}{file}'
        shutil.copy(futures_src / f'{pair_unified}{file}', fn_after)
    default_conf_usdt['datadir'] = tmp_path
    migrate_binance_futures_data(default_conf_usdt)
    for file in files:
        fn_after = futures_dst / f'{pair_unified}{file}'
        assert fn_after.exists()

@pytest.mark.usefixtures('init_persistence')
def test_binance_mig_db_conversion(default_conf_usdt, fee, caplog):
    if False:
        while True:
            i = 10
    migrate_binance_futures_names(default_conf_usdt)
    create_mock_trades_usdt(fee, None)
    for t in Trade.get_trades():
        t.trading_mode = 'FUTURES'
        t.exchange = 'binance'
    Trade.commit()
    default_conf_usdt['trading_mode'] = 'futures'
    migrate_binance_futures_names(default_conf_usdt)
    assert log_has('Migrating binance futures pairs in database.', caplog)