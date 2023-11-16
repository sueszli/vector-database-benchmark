from datetime import datetime, timedelta, timezone
import pytest
from freqtrade.persistence import PairLocks
from freqtrade.persistence.models import PairLock
from freqtrade.util import dt_now

@pytest.mark.parametrize('use_db', (False, True))
@pytest.mark.usefixtures('init_persistence')
def test_PairLocks(use_db):
    if False:
        for i in range(10):
            print('nop')
    PairLocks.timeframe = '5m'
    PairLocks.use_db = use_db
    if use_db:
        assert len(PairLock.get_all_locks().all()) == 0
    assert PairLocks.use_db == use_db
    pair = 'ETH/BTC'
    assert not PairLocks.is_pair_locked(pair)
    PairLocks.lock_pair(pair, dt_now() + timedelta(minutes=4))
    assert PairLocks.is_pair_locked(pair)
    assert PairLocks.is_pair_locked(pair, side='long')
    assert PairLocks.is_pair_locked(pair, side='short')
    pair = 'BNB/BTC'
    PairLocks.lock_pair(pair, dt_now() + timedelta(minutes=4), side='long')
    assert not PairLocks.is_pair_locked(pair)
    assert PairLocks.is_pair_locked(pair, side='long')
    assert not PairLocks.is_pair_locked(pair, side='short')
    pair = 'BNB/USDT'
    PairLocks.lock_pair(pair, dt_now() + timedelta(minutes=4), side='short')
    assert not PairLocks.is_pair_locked(pair)
    assert not PairLocks.is_pair_locked(pair, side='long')
    assert PairLocks.is_pair_locked(pair, side='short')
    pair = 'XRP/BTC'
    assert not PairLocks.is_pair_locked(pair)
    PairLocks.unlock_pair(pair)
    PairLocks.lock_pair(pair, dt_now() + timedelta(minutes=4))
    assert PairLocks.is_pair_locked(pair)
    locks = PairLocks.get_pair_locks(None)
    assert len(locks) == 2
    pair = 'ETH/BTC'
    PairLocks.unlock_pair(pair)
    assert not PairLocks.is_pair_locked(pair)
    assert not PairLocks.is_global_lock()
    pair = 'BTC/USDT'
    lock_time = datetime(2020, 5, 1, 14, 30, 0, tzinfo=timezone.utc)
    PairLocks.lock_pair(pair, lock_time)
    assert not PairLocks.is_pair_locked(pair)
    assert PairLocks.is_pair_locked(pair, lock_time + timedelta(minutes=-10))
    assert not PairLocks.is_global_lock(lock_time + timedelta(minutes=-10))
    assert PairLocks.is_pair_locked(pair, lock_time + timedelta(minutes=-50))
    assert not PairLocks.is_global_lock(lock_time + timedelta(minutes=-50))
    assert not PairLocks.is_pair_locked(pair, lock_time + timedelta(minutes=10))
    locks = PairLocks.get_pair_locks(pair, lock_time + timedelta(minutes=-2))
    assert len(locks) == 1
    assert 'PairLock' in str(locks[0])
    PairLocks.unlock_pair(pair, lock_time + timedelta(minutes=-2))
    assert not PairLocks.is_global_lock(lock_time + timedelta(minutes=-50))
    PairLocks.lock_pair('*', lock_time)
    assert PairLocks.is_global_lock(lock_time + timedelta(minutes=-50))
    assert PairLocks.is_pair_locked(pair, lock_time + timedelta(minutes=-50))
    assert PairLocks.is_pair_locked('XRP/USDT', lock_time + timedelta(minutes=-50))
    if use_db:
        locks = PairLocks.get_all_locks()
        locks_db = PairLock.get_all_locks().all()
        assert len(locks) == len(locks_db)
        assert len(locks_db) > 0
    else:
        assert len(PairLocks.get_all_locks()) > 0
        assert len(PairLock.get_all_locks().all()) == 0
    PairLocks.reset_locks()
    PairLocks.use_db = True

@pytest.mark.parametrize('use_db', (False, True))
@pytest.mark.usefixtures('init_persistence')
def test_PairLocks_getlongestlock(use_db):
    if False:
        i = 10
        return i + 15
    PairLocks.timeframe = '5m'
    PairLocks.use_db = use_db
    if use_db:
        assert len(PairLock.get_all_locks().all()) == 0
    assert PairLocks.use_db == use_db
    pair = 'ETH/BTC'
    assert not PairLocks.is_pair_locked(pair)
    PairLocks.lock_pair(pair, dt_now() + timedelta(minutes=4))
    assert PairLocks.is_pair_locked(pair)
    lock = PairLocks.get_pair_longest_lock(pair)
    assert lock.lock_end_time.replace(tzinfo=timezone.utc) > dt_now() + timedelta(minutes=3)
    assert lock.lock_end_time.replace(tzinfo=timezone.utc) < dt_now() + timedelta(minutes=14)
    PairLocks.lock_pair(pair, dt_now() + timedelta(minutes=15))
    assert PairLocks.is_pair_locked(pair)
    lock = PairLocks.get_pair_longest_lock(pair)
    assert lock.lock_end_time.replace(tzinfo=timezone.utc) > dt_now() + timedelta(minutes=14)
    PairLocks.reset_locks()
    PairLocks.use_db = True

@pytest.mark.parametrize('use_db', (False, True))
@pytest.mark.usefixtures('init_persistence')
def test_PairLocks_reason(use_db):
    if False:
        while True:
            i = 10
    PairLocks.timeframe = '5m'
    PairLocks.use_db = use_db
    if use_db:
        assert len(PairLock.get_all_locks().all()) == 0
    assert PairLocks.use_db == use_db
    PairLocks.lock_pair('XRP/USDT', dt_now() + timedelta(minutes=4), 'TestLock1')
    PairLocks.lock_pair('ETH/USDT', dt_now() + timedelta(minutes=4), 'TestLock2')
    assert PairLocks.is_pair_locked('XRP/USDT')
    assert PairLocks.is_pair_locked('ETH/USDT')
    PairLocks.unlock_reason('TestLock1')
    assert not PairLocks.is_pair_locked('XRP/USDT')
    assert PairLocks.is_pair_locked('ETH/USDT')
    PairLocks.reset_locks()
    PairLocks.use_db = True