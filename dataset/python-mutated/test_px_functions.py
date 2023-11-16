import plotly.express as px
import plotly.graph_objects as go
from numpy.testing import assert_array_equal
import numpy as np
import pandas as pd
import pytest

def _compare_figures(go_trace, px_fig):
    if False:
        return 10
    'Compare a figure created with a go trace and a figure created with\n    a px function call. Check that all values inside the go Figure are the\n    same in the px figure (which sets more parameters).\n    '
    go_fig = go.Figure(go_trace)
    go_fig = go_fig.to_plotly_json()
    px_fig = px_fig.to_plotly_json()
    del go_fig['layout']['template']
    del px_fig['layout']['template']
    for key in go_fig['data'][0]:
        assert_array_equal(go_fig['data'][0][key], px_fig['data'][0][key])
    for key in go_fig['layout']:
        assert go_fig['layout'][key] == px_fig['layout'][key]

def test_pie_like_px():
    if False:
        i = 10
        return i + 15
    labels = ['Oxygen', 'Hydrogen', 'Carbon_Dioxide', 'Nitrogen']
    values = [4500, 2500, 1053, 500]
    fig = px.pie(names=labels, values=values)
    trace = go.Pie(labels=labels, values=values)
    _compare_figures(trace, fig)
    labels = ['Eve', 'Cain', 'Seth', 'Enos', 'Noam', 'Abel', 'Awan', 'Enoch', 'Azura']
    parents = ['', 'Eve', 'Eve', 'Seth', 'Seth', 'Eve', 'Eve', 'Awan', 'Eve']
    values = [10, 14, 12, 10, 2, 6, 6, 4, 4]
    fig = px.sunburst(names=labels, parents=parents, values=values)
    trace = go.Sunburst(labels=labels, parents=parents, values=values)
    _compare_figures(trace, fig)
    fig = px.treemap(names=labels, parents=parents, values=values)
    trace = go.Treemap(labels=labels, parents=parents, values=values)
    _compare_figures(trace, fig)
    x = ['A', 'B', 'C']
    y = [3, 2, 1]
    fig = px.funnel(y=y, x=x)
    trace = go.Funnel(y=y, x=x)
    _compare_figures(trace, fig)
    fig = px.funnel_area(values=y, names=x)
    trace = go.Funnelarea(values=y, labels=x)
    _compare_figures(trace, fig)

def test_sunburst_treemap_colorscales():
    if False:
        return 10
    labels = ['Eve', 'Cain', 'Seth', 'Enos', 'Noam', 'Abel', 'Awan', 'Enoch', 'Azura']
    parents = ['', 'Eve', 'Eve', 'Seth', 'Seth', 'Eve', 'Eve', 'Awan', 'Eve']
    values = [10, 14, 12, 10, 2, 6, 6, 4, 4]
    for (func, colorway) in zip([px.sunburst, px.treemap], ['sunburstcolorway', 'treemapcolorway']):
        fig = func(names=labels, parents=parents, values=values, color=values, color_continuous_scale='Viridis', range_color=(5, 15))
        assert fig.layout.coloraxis.cmin, fig.layout.coloraxis.cmax == (5, 15)
        color_seq = px.colors.sequential.Reds
        fig = func(names=labels, parents=parents, values=values, color=labels, color_discrete_sequence=color_seq)
        assert np.all([col in color_seq for col in fig.data[0].marker.colors])
        fig = func(names=labels, parents=parents, values=values, color=values)
        assert [el[0] == px.colors.sequential.Viridis for (i, el) in enumerate(fig.layout.coloraxis.colorscale)]
        fig = func(names=labels, parents=parents, values=values, color=values, color_discrete_sequence=color_seq)
        assert [el[0] == px.colors.sequential.Viridis for (i, el) in enumerate(fig.layout.coloraxis.colorscale)]
        color_seq = px.colors.sequential.Reds
        fig = func(names=labels, parents=parents, values=values, color_discrete_sequence=color_seq)
        assert list(fig.layout[colorway]) == color_seq

def test_sunburst_treemap_with_path():
    if False:
        for i in range(10):
            print('nop')
    vendors = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    sectors = ['Tech', 'Tech', 'Finance', 'Finance', 'Tech', 'Tech', 'Finance', 'Finance']
    regions = ['North', 'North', 'North', 'North', 'South', 'South', 'South', 'South']
    values = [1, 3, 2, 4, 2, 2, 1, 4]
    total = ['total'] * 8
    df = pd.DataFrame(dict(vendors=vendors, sectors=sectors, regions=regions, values=values, total=total))
    path = ['total', 'regions', 'sectors', 'vendors']
    fig = px.sunburst(df, path=path)
    assert fig.data[0].branchvalues == 'total'
    fig = px.sunburst(df, path=path, values='values')
    assert fig.data[0].branchvalues == 'total'
    assert fig.data[0].values[-1] == np.sum(values)
    fig = px.sunburst(df, path=path, values='values')
    assert fig.data[0].branchvalues == 'total'
    assert fig.data[0].values[-1] == np.sum(values)
    df['values'] = ['1 000', '3 000', '2', '4', '2', '2', '1 000', '4 000']
    msg = 'Column `values` of `df` could not be converted to a numerical data type.'
    with pytest.raises(ValueError, match=msg):
        fig = px.sunburst(df, path=path, values='values')
    path = [df.total, 'regions', df.sectors, 'vendors']
    fig = px.sunburst(df, path=path)
    assert fig.data[0].branchvalues == 'total'
    df['values'] = 1
    fig = px.sunburst(df, path=path, values='values', color='values')
    assert 'coloraxis' in fig.data[0].marker
    assert np.all(np.array(fig.data[0].marker.colors) == 1)
    assert fig.data[0].values[-1] == 8

def test_sunburst_treemap_with_path_and_hover():
    if False:
        return 10
    df = px.data.tips()
    fig = px.sunburst(df, path=['sex', 'day', 'time', 'smoker'], color='smoker', hover_data=['smoker'])
    assert 'smoker' in fig.data[0].hovertemplate
    df = px.data.gapminder().query('year == 2007')
    fig = px.sunburst(df, path=['continent', 'country'], color='lifeExp', hover_data=df.columns)
    assert fig.layout.coloraxis.colorbar.title.text == 'lifeExp'
    df = px.data.tips()
    fig = px.sunburst(df, path=['sex', 'day', 'time', 'smoker'], hover_name='smoker')
    assert 'smoker' not in fig.data[0].hovertemplate
    assert '%{hovertext}' in fig.data[0].hovertemplate
    df = px.data.tips()
    fig = px.sunburst(df, path=['sex', 'day', 'time', 'smoker'], custom_data=['smoker'])
    assert fig.data[0].customdata[0][0] in ['Yes', 'No']
    assert 'smoker' not in fig.data[0].hovertemplate
    assert '%{hovertext}' not in fig.data[0].hovertemplate

def test_sunburst_treemap_with_path_color():
    if False:
        i = 10
        return i + 15
    vendors = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    sectors = ['Tech', 'Tech', 'Finance', 'Finance', 'Tech', 'Tech', 'Finance', 'Finance']
    regions = ['North', 'North', 'North', 'North', 'South', 'South', 'South', 'South']
    values = [1, 3, 2, 4, 2, 2, 1, 4]
    calls = [8, 2, 1, 3, 2, 2, 4, 1]
    total = ['total'] * 8
    df = pd.DataFrame(dict(vendors=vendors, sectors=sectors, regions=regions, values=values, total=total, calls=calls))
    path = ['total', 'regions', 'sectors', 'vendors']
    fig = px.sunburst(df, path=path, values='values', color='calls')
    colors = fig.data[0].marker.colors
    assert np.all(np.array(colors[:8]) == np.array(calls))
    fig = px.sunburst(df, path=path, color='calls')
    colors = fig.data[0].marker.colors
    assert np.all(np.array(colors[:8]) == np.array(calls))
    df['hover'] = [el.lower() for el in vendors]
    fig = px.sunburst(df, path=path, color='calls', hover_data=['hover'])
    custom = fig.data[0].customdata
    assert np.all(custom[:8, 0] == df['hover'])
    assert np.all(custom[8:, 0] == '(?)')
    assert np.all(custom[:8, 1] == df['calls'])
    fig = px.sunburst(df, path=path, color='vendors')
    assert len(np.unique(fig.data[0].marker.colors)) == 9
    cmap = {'Tech': 'yellow', 'Finance': 'magenta', '(?)': 'black'}
    fig = px.sunburst(df, path=path, color='sectors', color_discrete_map=cmap)
    assert np.all(np.in1d(fig.data[0].marker.colors, list(cmap.values())))
    df['regions'] = df['regions'].map({'North': 1, 'South': 2})
    path = ['total', 'regions', 'sectors', 'vendors']
    fig = px.sunburst(df, path=path, values='values', color='calls')
    colors = fig.data[0].marker.colors
    assert np.all(np.array(colors[:8]) == np.array(calls))

def test_sunburst_treemap_column_parent():
    if False:
        for i in range(10):
            print('nop')
    vendors = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    sectors = ['Tech', 'Tech', 'Finance', 'Finance', 'Tech', 'Tech', 'Finance', 'Finance']
    regions = ['North', 'North', 'North', 'North', 'South', 'South', 'South', 'South']
    values = [1, 3, 2, 4, 2, 2, 1, 4]
    df = pd.DataFrame(dict(id=vendors, sectors=sectors, parent=regions, values=values))
    path = ['parent', 'sectors', 'id']
    px.sunburst(df, path=path, values='values')

def test_sunburst_treemap_with_path_non_rectangular():
    if False:
        i = 10
        return i + 15
    vendors = ['A', 'B', 'C', 'D', None, 'E', 'F', 'G', 'H', None]
    sectors = ['Tech', 'Tech', 'Finance', 'Finance', None, 'Tech', 'Tech', 'Finance', 'Finance', 'Finance']
    regions = ['North', 'North', 'North', 'North', 'North', 'South', 'South', 'South', 'South', 'South']
    values = [1, 3, 2, 4, 1, 2, 2, 1, 4, 1]
    total = ['total'] * 10
    df = pd.DataFrame(dict(vendors=vendors, sectors=sectors, regions=regions, values=values, total=total))
    path = ['total', 'regions', 'sectors', 'vendors']
    msg = 'Non-leaves rows are not permitted in the dataframe'
    with pytest.raises(ValueError, match=msg):
        fig = px.sunburst(df, path=path, values='values')
    df.loc[df['vendors'].isnull(), 'sectors'] = 'Other'
    fig = px.sunburst(df, path=path, values='values')
    assert fig.data[0].values[-1] == np.sum(values)

def test_pie_funnelarea_colorscale():
    if False:
        while True:
            i = 10
    labels = ['A', 'B', 'C', 'D']
    values = [3, 2, 1, 4]
    for (func, colorway) in zip([px.sunburst, px.treemap], ['sunburstcolorway', 'treemapcolorway']):
        color_seq = px.colors.sequential.Reds
        fig = func(names=labels, values=values, color_discrete_sequence=color_seq)
        assert list(fig.layout[colorway]) == color_seq
        color_seq = px.colors.sequential.Reds
        fig = func(names=labels, values=values, color=labels, color_discrete_sequence=color_seq)
        assert np.all([col in color_seq for col in fig.data[0].marker.colors])

def test_funnel():
    if False:
        while True:
            i = 10
    fig = px.funnel(x=[5, 4, 3, 3, 2, 1], y=['A', 'B', 'C', 'A', 'B', 'C'], color=['0', '0', '0', '1', '1', '1'])
    assert len(fig.data) == 2

def test_parcats_dimensions_max():
    if False:
        print('Hello World!')
    df = px.data.tips()
    fig = px.parallel_categories(df)
    assert [d.label for d in fig.data[0].dimensions] == ['sex', 'smoker', 'day', 'time', 'size']
    fig = px.parallel_categories(df, dimensions=['sex', 'smoker', 'day'])
    assert [d.label for d in fig.data[0].dimensions] == ['sex', 'smoker', 'day']
    fig = px.parallel_categories(df, dimensions_max_cardinality=4)
    assert [d.label for d in fig.data[0].dimensions] == ['sex', 'smoker', 'day', 'time']
    fig = px.parallel_categories(df, dimensions=['sex', 'smoker', 'day', 'size'], dimensions_max_cardinality=4)
    assert [d.label for d in fig.data[0].dimensions] == ['sex', 'smoker', 'day', 'size']

@pytest.mark.parametrize('histfunc,y', [(None, None), ('count', 'tip')])
def test_histfunc_hoverlabels_univariate(histfunc, y):
    if False:
        return 10

    def check_label(label, fig):
        if False:
            return 10
        assert fig.layout.yaxis.title.text == label
        assert label + '=' in fig.data[0].hovertemplate
    df = px.data.tips()
    fig = px.histogram(df, x='total_bill', y=y, histfunc=histfunc)
    check_label('count', fig)
    for histnorm in ['probability', 'percent', 'density', 'probability density']:
        fig = px.histogram(df, x='total_bill', y=y, histfunc=histfunc, histnorm=histnorm)
        check_label(histnorm, fig)
    for histnorm in ['probability', 'percent', 'density', 'probability density']:
        for barnorm in ['percent', 'fraction']:
            fig = px.histogram(df, x='total_bill', y=y, histfunc=histfunc, histnorm=histnorm, barnorm=barnorm)
            check_label('%s (normalized as %s)' % (histnorm, barnorm), fig)

def test_histfunc_hoverlabels_bivariate():
    if False:
        return 10

    def check_label(label, fig):
        if False:
            print('Hello World!')
        assert fig.layout.yaxis.title.text == label
        assert label + '=' in fig.data[0].hovertemplate
    df = px.data.tips()
    fig = px.histogram(df, x='total_bill', y='tip')
    check_label('sum of tip', fig)
    fig = px.histogram(df, x='total_bill', y='tip', histnorm='probability')
    check_label('fraction of sum of tip', fig)
    fig = px.histogram(df, x='total_bill', y='tip', histnorm='percent')
    check_label('percent of sum of tip', fig)
    for histnorm in ['density', 'probability density']:
        fig = px.histogram(df, x='total_bill', y='tip', histnorm=histnorm)
        check_label('%s weighted by tip' % histnorm, fig)
    for histnorm in ['density', 'probability density']:
        for barnorm in ['fraction', 'percent']:
            fig = px.histogram(df, x='total_bill', y='tip', histnorm=histnorm, barnorm=barnorm)
            check_label('%s weighted by tip (normalized as %s)' % (histnorm, barnorm), fig)
    fig = px.histogram(df, x='total_bill', y='tip', histfunc='min', histnorm='probability', barnorm='percent')
    check_label('fraction of sum of min of tip (normalized as percent)', fig)
    fig = px.histogram(df, x='total_bill', y='tip', histfunc='avg', histnorm='percent', barnorm='fraction')
    check_label('percent of sum of avg of tip (normalized as fraction)', fig)
    fig = px.histogram(df, x='total_bill', y='tip', histfunc='max', histnorm='density')
    check_label('density of max of tip', fig)

def test_timeline():
    if False:
        print('Hello World!')
    df = pd.DataFrame([dict(Task='Job A', Start='2009-01-01', Finish='2009-02-28'), dict(Task='Job B', Start='2009-03-05', Finish='2009-04-15'), dict(Task='Job C', Start='2009-02-20', Finish='2009-05-30')])
    fig = px.timeline(df, x_start='Start', x_end='Finish', y='Task', color='Task')
    assert len(fig.data) == 3
    assert fig.layout.xaxis.type == 'date'
    assert fig.layout.xaxis.title.text is None
    fig = px.timeline(df, x_start='Start', x_end='Finish', y='Task', facet_row='Task')
    assert len(fig.data) == 3
    assert fig.data[1].xaxis == 'x2'
    assert fig.layout.xaxis.type == 'date'
    msg = 'Both x_start and x_end are required'
    with pytest.raises(ValueError, match=msg):
        px.timeline(df, x_start='Start', y='Task', color='Task')
    msg = 'Both x_start and x_end must refer to data convertible to datetimes.'
    with pytest.raises(TypeError, match=msg):
        px.timeline(df, x_start='Start', x_end=['a', 'b', 'c'], y='Task', color='Task')