"""
Tests for the IPv8Monitor.
By design, these tests only test if the IPv8 walk_interval is appropriately scaled.
These tests do not test how much the walk_interval is scaled.
"""
import threading
import time
import pytest
from ipv8.taskmanager import TaskManager
from tribler.core.components.ipv8.ipv8_health_monitor import IPv8Monitor
DEFAULT_WALK_INTERVAL = 0.5

@pytest.fixture(name='ipv8')
def fixture_ipv8():
    if False:
        print('Hello World!')
    return type('IPv8', (object,), {'strategies': [], 'overlay_lock': threading.RLock(), 'walk_interval': DEFAULT_WALK_INTERVAL})

@pytest.fixture(name='task_manager')
async def fixture_task_manager():
    task_manager = TaskManager()
    yield task_manager
    await task_manager.shutdown_task_manager()

@pytest.fixture(name='ipv8_health_monitor')
def fixture_ipv8_health_monitor(ipv8):
    if False:
        print('Hello World!')
    return IPv8Monitor(ipv8, DEFAULT_WALK_INTERVAL, 3.0, 0.01)

async def test_start(task_manager, ipv8_health_monitor):
    mock_interval = 7.7
    ipv8_health_monitor.start(task_manager, mock_interval)
    assert len(task_manager.get_tasks()) == 1
    assert ipv8_health_monitor.interval == mock_interval

def test_choke_exceed(ipv8_health_monitor):
    if False:
        for i in range(10):
            print('nop')
    '\n    We should slow down, if choke is detected.\n    '
    ipv8_health_monitor.measurement_strategy.history = [(time.time(), 2 * ipv8_health_monitor.choke_limit)]
    ipv8_health_monitor.last_check = 0
    ipv8_health_monitor.auto_scale_ipv8()
    assert ipv8_health_monitor.ipv8_instance.walk_interval > DEFAULT_WALK_INTERVAL

def test_choke_exceed_maximum(ipv8_health_monitor):
    if False:
        print('Hello World!')
    '\n    We should not change, if choke is detected and we are already at the slowest speed.\n    '
    ipv8_health_monitor.measurement_strategy.history = [(time.time(), 2 * ipv8_health_monitor.choke_limit)]
    ipv8_health_monitor.last_check = 0
    ipv8_health_monitor.current_rate = ipv8_health_monitor.max_update_rate
    ipv8_health_monitor.auto_scale_ipv8()
    assert ipv8_health_monitor.ipv8_instance.walk_interval == ipv8_health_monitor.max_update_rate

def test_no_choke(ipv8_health_monitor):
    if False:
        while True:
            i = 10
    "\n    We should speed up our walk_interval if we're not choked.\n    "
    ipv8_health_monitor.measurement_strategy.history = [(time.time(), ipv8_health_monitor.choke_limit)]
    ipv8_health_monitor.last_check = 0
    ipv8_health_monitor.current_rate = ipv8_health_monitor.max_update_rate
    ipv8_health_monitor.auto_scale_ipv8()
    assert ipv8_health_monitor.ipv8_instance.walk_interval < ipv8_health_monitor.max_update_rate

def test_no_choke_minimum(ipv8_health_monitor):
    if False:
        for i in range(10):
            print('nop')
    "\n    We should not change our walk_interval if we're already at the minimum.\n    "
    ipv8_health_monitor.measurement_strategy.history = [(time.time(), ipv8_health_monitor.choke_limit)]
    ipv8_health_monitor.last_check = 0
    ipv8_health_monitor.current_rate = ipv8_health_monitor.min_update_rate
    ipv8_health_monitor.auto_scale_ipv8()
    assert ipv8_health_monitor.ipv8_instance.walk_interval == DEFAULT_WALK_INTERVAL

def test_intialize_minimum(ipv8_health_monitor):
    if False:
        for i in range(10):
            print('nop')
    "\n    We should not deviate from the minimum update rate if we don't have a history.\n    "
    ipv8_health_monitor.measurement_strategy.history = []
    ipv8_health_monitor.last_check = 0
    ipv8_health_monitor.auto_scale_ipv8()
    assert ipv8_health_monitor.ipv8_instance.walk_interval == DEFAULT_WALK_INTERVAL

def test_update_rate(ipv8_health_monitor):
    if False:
        print('Hello World!')
    '\n    We should not update our rate when the last check was within the interval.\n    '
    ipv8_health_monitor.measurement_strategy.history = [(time.time(), ipv8_health_monitor.choke_limit)]
    ipv8_health_monitor.current_rate = ipv8_health_monitor.max_update_rate
    ipv8_health_monitor.ipv8_instance.walk_interval = ipv8_health_monitor.max_update_rate
    ipv8_health_monitor.auto_scale_ipv8()
    assert ipv8_health_monitor.ipv8_instance.walk_interval == ipv8_health_monitor.max_update_rate