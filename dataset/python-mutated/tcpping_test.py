from scylla.tcpping import Ping, ping

def test_ping_class():
    if False:
        return 10
    p = Ping('www.example.com', port=80)
    p.ping(5, 0.01)
    assert p.get_maximum() >= p.get_average()
    assert p.get_minimum() <= p.get_average()
    assert p.get_average() >= 0
    assert p.get_success_rate() >= 0.0

def test_ping_func():
    if False:
        print('Hello World!')
    (avg, rate) = ping('www.example.com', 80)
    assert avg > 0
    assert rate > 0.0