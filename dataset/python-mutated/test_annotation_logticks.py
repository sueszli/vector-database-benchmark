import numpy as np
import pandas as pd
import pytest
from mizani.transforms import label_log, log_trans
from plotnine import aes, annotation_logticks, coord_flip, element_line, facet_wrap, geom_point, ggplot, scale_x_continuous, scale_x_log10, scale_y_log10, theme
from plotnine.exceptions import PlotnineWarning
data = pd.DataFrame({'x': 10 ** np.arange(4)})

def test_annotation_logticks():
    if False:
        for i in range(10):
            print('nop')
    p = ggplot(data, aes('x', 'x')) + annotation_logticks(sides='b', size=0.75) + geom_point() + scale_x_log10() + scale_y_log10() + theme(panel_grid_minor=element_line(color='green'), panel_grid_major=element_line(color='red'))
    assert p == 'annotation_logticks'

def test_annotation_logticks_faceting():
    if False:
        for i in range(10):
            print('nop')
    n = len(data)
    data2 = pd.DataFrame({'x': np.hstack([data['x'], data['x']]), 'g': list('a' * n + 'b' * n)})
    p = ggplot(data2) + annotation_logticks(sides='b', size=0.75) + geom_point(aes('x', 'x')) + scale_x_log10() + scale_y_log10() + facet_wrap('g') + theme(panel_grid_minor=element_line(color='green'), panel_grid_major=element_line(color='red'))
    assert p == 'annotation_logticks_faceting'

def test_annotation_logticks_coord_flip():
    if False:
        print('Hello World!')
    p = ggplot(data, aes('x', 'x')) + annotation_logticks(sides='b', size=0.75) + geom_point() + scale_x_log10() + scale_y_log10() + coord_flip() + theme(panel_grid_minor=element_line(color='green'), panel_grid_major=element_line(color='red'))
    assert p == 'annotation_logticks_coord_flip'

def test_annotation_logticks_coord_flip_discrete():
    if False:
        i = 10
        return i + 15
    data = pd.DataFrame({'x': 10.0 ** (np.arange(4) - 1)})
    data2 = data.assign(discrete=pd.Categorical(['A' + str(int(a)) for a in data['x']]))
    data2 = data2.drop(data2.index[1:3])
    p = ggplot(data2, aes('discrete', 'x')) + annotation_logticks(sides='l', size=0.75) + geom_point() + scale_y_log10() + coord_flip() + theme(panel_grid_minor=element_line(color='green'), panel_grid_major=element_line(color='red'))
    assert p == 'annotation_logticks_coord_flip_discrete'

def test_annotation_logticks_coord_flip_discrete_bottom():
    if False:
        return 10
    data = pd.DataFrame({'x': 10.0 ** (np.arange(4) - 1)})
    data2 = data.assign(discrete=pd.Categorical(['A' + str(int(a)) for a in data['x']]))
    data2 = data2.drop(data2.index[1:3])
    p = ggplot(data2, aes('x', 'discrete')) + annotation_logticks(sides='b', size=0.75) + geom_point() + scale_x_log10() + coord_flip() + theme(panel_grid_minor=element_line(color='green'), panel_grid_major=element_line(color='red'))
    assert p == 'annotation_logticks_coord_flip_discrete_bottom'

def test_annotation_logticks_base_8():
    if False:
        i = 10
        return i + 15
    base = 8
    data = pd.DataFrame({'x': base ** np.arange(4)})
    p = ggplot(data, aes('x', 'x')) + annotation_logticks(sides='b', size=0.75) + geom_point() + scale_x_continuous(trans=log_trans(base=base), labels=label_log(base=base, mathtex=True)) + theme(panel_grid_minor=element_line(color='green'), panel_grid_major=element_line(color='red'))
    assert p == 'annotation_logticks_base_8'

def test_annotation_logticks_base_5():
    if False:
        i = 10
        return i + 15
    base = 5
    data = pd.DataFrame({'x': base ** np.arange(4)})
    p = ggplot(data, aes('x', 'x')) + annotation_logticks(sides='b', size=0.75) + geom_point() + scale_x_continuous(trans=log_trans(base=base)) + theme(panel_grid_minor=element_line(color='green'), panel_grid_major=element_line(color='red'))
    assert p == 'annotation_logticks_base_5'

def test_wrong_bases():
    if False:
        i = 10
        return i + 15
    p = ggplot(data, aes('x', 'x')) + annotation_logticks(sides='b', size=0.75, base=10) + geom_point()
    with pytest.warns(PlotnineWarning):
        p.draw_test()
    p = ggplot(data, aes('x', 'x')) + annotation_logticks(sides='b', size=0.75, base=10) + scale_x_continuous(trans=log_trans(8)) + geom_point()
    with pytest.warns(PlotnineWarning):
        p.draw_test()
    data2 = data.assign(discrete=pd.Categorical([str(a) for a in data['x']]))
    p = ggplot(data2, aes('discrete', 'x')) + annotation_logticks(sides='b', size=0.75, base=None) + geom_point()
    with pytest.warns(PlotnineWarning):
        p.draw_test()
    data2 = data.assign(discrete=pd.Categorical([str(a) for a in data['x']]))
    p = ggplot(data2, aes('x', 'discrete')) + annotation_logticks(sides='l', size=0.75, base=None) + geom_point()
    with pytest.warns(PlotnineWarning):
        p.draw_test()
    data2 = data.assign(discrete=pd.Categorical([str(a) for a in data['x']]))
    p = ggplot(data2, aes('discrete', 'x')) + annotation_logticks(sides='b', size=0.75, base=None) + geom_point() + coord_flip()
    with pytest.warns(PlotnineWarning):
        p.draw_test()
    data2 = data.assign(discrete=pd.Categorical([str(a) for a in data['x']]))
    p = ggplot(data2, aes('x', 'discrete')) + annotation_logticks(sides='l', size=0.75, base=None) + geom_point() + coord_flip()
    with pytest.warns(PlotnineWarning):
        p.draw_test()