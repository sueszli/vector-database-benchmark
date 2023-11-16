from plotnine import aes, geom_blank, ggplot
from plotnine.data import mtcars

def test_blank():
    if False:
        i = 10
        return i + 15
    gg = ggplot(mtcars, aes(x='wt', y='mpg'))
    gg = gg + geom_blank()
    assert gg == 'blank'