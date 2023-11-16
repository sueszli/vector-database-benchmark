from libqtile import widget
RESPONSE = 'London: +17°C'

def test_wttr_methods():
    if False:
        while True:
            i = 10
    wttr = widget.Wttr(location={'London': 'Home'})
    assert wttr._get_url() == 'https://wttr.in/London?m&format=3&lang=en'
    assert wttr.parse(RESPONSE) == 'Home: +17°C'

def test_wttr_no_location():
    if False:
        for i in range(10):
            print('nop')
    wttr = widget.Wttr()
    assert wttr._get_url() == 'https://wttr.in/?m&format=3&lang=en'