import pytest
from patterns.behavioral.state import Radio

@pytest.fixture
def radio():
    if False:
        for i in range(10):
            print('nop')
    return Radio()

def test_initial_state(radio):
    if False:
        i = 10
        return i + 15
    assert radio.state.name == 'AM'

def test_initial_am_station(radio):
    if False:
        print('Hello World!')
    initial_pos = radio.state.pos
    assert radio.state.stations[initial_pos] == '1250'

def test_toggle_amfm(radio):
    if False:
        i = 10
        return i + 15
    assert radio.state.name == 'AM'
    radio.toggle_amfm()
    assert radio.state.name == 'FM'
    radio.toggle_amfm()
    assert radio.state.name == 'AM'