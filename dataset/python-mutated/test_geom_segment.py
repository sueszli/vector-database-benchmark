import pandas as pd
from plotnine import aes, arrow, geom_segment, ggplot
n = 4
data = pd.DataFrame({'x': range(1, n + 1), 'xend': range(2, n + 2), 'y': range(n, 0, -1), 'yend': range(n, 0, -1), 'z': range(1, n + 1)})

def test_aesthetics():
    if False:
        print('Hello World!')
    p = ggplot(data, aes('x', 'y', xend='xend', yend='yend')) + geom_segment(size=2) + geom_segment(aes(yend='yend+1', color='factor(z)'), size=2) + geom_segment(aes(yend='yend+2', linetype='factor(z)'), size=2) + geom_segment(aes(yend='yend+3', size='z'), show_legend=False) + geom_segment(aes(yend='yend+4', alpha='z'), size=2, show_legend=False)
    assert p == 'aesthetics'

def test_arrow():
    if False:
        while True:
            i = 10
    p = ggplot(data, aes('x', 'y', xend='xend', yend='yend')) + geom_segment(aes('x+2', xend='xend+2'), arrow=arrow(), size=2) + geom_segment(aes('x+4', xend='xend+4'), arrow=arrow(ends='first'), size=2) + geom_segment(aes('x+6', xend='xend+6'), arrow=arrow(ends='both'), size=2)
    assert p == 'arrow'