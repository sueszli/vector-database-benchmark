import pandas as pd
import pytest
from ydata_profiling import ProfileReport

def generate_cat_data_series(categories):
    if False:
        print('Hello World!')
    dummy_data = []
    for (cat, i) in categories.items():
        dummy_data.extend([cat] * i)
    return pd.DataFrame({'dummy_cat': dummy_data})
dummy_bool_data = generate_cat_data_series(pd.Series({True: 82, False: 36}))
dummy_cat_data = generate_cat_data_series(pd.Series({'Amadeou_plus': 75, 'Beta_front': 50, 'Calciumus': 20, 'Dimitrius': 1, 'esperagus_anonymoliumus': 75, 'FrigaTTTBrigde_Writap': 50, 'galgarartiy': 30, 'He': 1, 'I': 10, 'JimISGODDOT': 1}))

def generate_report(data):
    if False:
        for i in range(10):
            print('nop')
    return ProfileReport(df=data, progress_bar=False, samples=None, correlations=None, missing_diagrams=None, duplicates=None, interactions=None)

@pytest.mark.parametrize('data', [dummy_bool_data, dummy_cat_data], ids=['bool', 'cat'])
@pytest.mark.parametrize('plot_type', ['bar', 'pie'])
def test_deactivated_cat_frequency_plot(data, plot_type):
    if False:
        for i in range(10):
            print('nop')
    profile = generate_report(data)
    profile.config.plot.cat_freq.show = False
    profile.config.plot.cat_freq.type = plot_type
    html_report = profile.to_html()
    assert 'Common Values (Plot)' not in html_report

@pytest.mark.parametrize('data', [dummy_bool_data, dummy_cat_data], ids=['bool', 'cat'])
def test_cat_frequency_default_barh_plot(data):
    if False:
        return 10
    profile = generate_report(data)
    html_report = profile.to_html()
    assert 'Common Values (Plot)' in html_report

@pytest.mark.parametrize('data', [dummy_bool_data, dummy_cat_data], ids=['bool', 'cat'])
def test_cat_frequency_pie_plot(data):
    if False:
        i = 10
        return i + 15
    profile = generate_report(data)
    profile.config.plot.cat_freq.type = 'pie'
    html_report = profile.to_html()
    assert 'pie' in html_report

@pytest.mark.parametrize('plot_type', ['bar', 'pie'])
def test_max_nuique_smaller_than_unique_cats(plot_type):
    if False:
        while True:
            i = 10
    profile = generate_report(dummy_cat_data)
    profile.config.plot.cat_freq.max_unique = 2
    profile.config.plot.cat_freq.type = plot_type
    html_report = profile.to_html()
    assert 'Common Values (Plot)' not in html_report

@pytest.mark.parametrize('plot_type', ['bar', 'pie'])
def test_cat_frequency_with_custom_colors(plot_type):
    if False:
        for i in range(10):
            print('nop')
    test_data = generate_cat_data_series(pd.Series({'A': 10, 'B': 10, 'C': 10}))
    custom_colors = {'gold': '#ffd700', 'b': '#0000ff', '#FF796C': '#ff796c'}
    profile = generate_report(test_data)
    profile.config.plot.cat_freq.colors = list(custom_colors.keys())
    profile.config.plot.cat_freq.type = plot_type
    html_report = profile.to_html()
    for (c, hex_code) in custom_colors.items():
        assert f'fill: {hex_code}' in html_report, f'Missing color code of {c}'

def test_more_cats_than_colors():
    if False:
        i = 10
        return i + 15
    test_data = generate_cat_data_series(pd.Series({'A': 10, 'B': 10, 'C': 10, 'D': 10}))
    custom_colors = {'gold': '#ffd700', 'b': '#0000ff', '#FF796C': '#ff796c'}
    profile = generate_report(test_data)
    profile.config.plot.cat_freq.colors = list(custom_colors.keys())
    html_report = profile.to_html()
    assert 'Common Values (Plot)' in html_report

@pytest.mark.parametrize('data', [dummy_bool_data, dummy_cat_data], ids=['bool', 'cat'])
def test_exception_with_invalid_cat_freq_type(data):
    if False:
        i = 10
        return i + 15
    profile = generate_report(data)
    profile.config.plot.cat_freq.type = 'box'
    with pytest.raises(ValueError):
        profile.to_html()