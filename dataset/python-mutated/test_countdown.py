from datetime import datetime, timedelta
from libqtile import widget
td = timedelta(days=10, hours=10, minutes=10, seconds=10)

def test_countdown_formatting():
    if False:
        for i in range(10):
            print('nop')
    countdown = widget.Countdown(date=datetime.now() + td, format='{D}d {H}h {M}m')
    output = countdown.poll()
    assert output == '10d 10h 10m'