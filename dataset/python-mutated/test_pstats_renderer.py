import os
import time
from pathlib import Path
from pstats import Stats
from test.fake_time_util import fake_time
from typing import Any
import pytest
from pyinstrument import Profiler
from pyinstrument.renderers import PstatsRenderer

def a():
    if False:
        for i in range(10):
            print('nop')
    b()
    c()

def b():
    if False:
        return 10
    d()

def c():
    if False:
        return 10
    d()

def d():
    if False:
        for i in range(10):
            print('nop')
    e()

def e():
    if False:
        print('Hello World!')
    time.sleep(1)

@pytest.fixture(scope='module')
def profiler_session():
    if False:
        return 10
    with fake_time():
        profiler = Profiler()
        profiler.start()
        a()
        profiler.stop()
        return profiler.last_session

def test_pstats_renderer(profiler_session, tmp_path):
    if False:
        i = 10
        return i + 15
    fname = tmp_path / 'test.pstats'
    pstats_data = PstatsRenderer().render(profiler_session)
    with open(fname, 'wb') as fid:
        fid.write(pstats_data.encode(encoding='utf-8', errors='surrogateescape'))
    stats: Any = Stats(str(fname))
    assert stats.total_tt > 0
    d_key = [k for k in stats.stats.keys() if k[2] == 'd'][0]
    d_val = stats.stats[d_key]
    d_cumtime = d_val[3]
    assert d_cumtime == pytest.approx(2)
    b_key = [k for k in stats.stats.keys() if k[2] == 'b'][0]
    c_key = [k for k in stats.stats.keys() if k[2] == 'c'][0]
    d_callers = d_val[4]
    b_cumtime = d_callers[b_key][3]
    c_cumtime = d_callers[c_key][3]
    assert b_cumtime == pytest.approx(1)
    assert c_cumtime == pytest.approx(1)
    e_key = [k for k in stats.stats.keys() if k[2] == 'e'][0]
    e_val = stats.stats[e_key]
    e_cumtime = e_val[3]
    assert e_cumtime == pytest.approx(2)

def test_round_trip_encoding_of_binary_data(tmp_path: Path):
    if False:
        return 10
    data_blob = os.urandom(1024)
    file = tmp_path / 'file.dat'
    data_blob_string = data_blob.decode(encoding='utf-8', errors='surrogateescape')
    with open(file, mode='w', encoding='utf-8', errors='surrogateescape', newline='') as f:
        f.write(data_blob_string)
    assert data_blob == data_blob_string.encode(encoding='utf-8', errors='surrogateescape')
    assert data_blob == file.read_bytes()