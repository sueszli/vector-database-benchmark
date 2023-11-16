import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import pytest
from plotnine import aes, geom_text, ggplot, ggtitle, theme
from plotnine.data import mtcars
from plotnine.exceptions import PlotnineError, PlotnineWarning
from plotnine.ggplot import save_as_pdf_pages

def p(N=3):
    if False:
        print('Hello World!')
    'Return *N* distinct plot objects.'
    template = ggplot(mtcars, aes(x='wt', y='mpg', label='name')) + geom_text()
    for i in range(1, N + 1):
        yield (template + ggtitle('%d of %d' % (i, N)))

def sequential_filenames():
    if False:
        print('Hello World!')
    '\n    Generate filenames for the tests\n    '
    for i in range(100):
        yield Path(f'filename-{i}.png')
filename_gen = sequential_filenames()

def assert_exist_and_clean(filename, msg=None):
    if False:
        for i in range(10):
            print('nop')
    if not msg:
        msg = f'File {filename} does not exist'
    assert filename.exists(), msg
    filename.unlink()

class TestArguments:

    def test_default_filename(self):
        if False:
            i = 10
            return i + 15
        plots = list(p())
        save_as_pdf_pages(plots, verbose=False)
        fn = plots[0]._save_filename('pdf')
        assert_exist_and_clean(fn, 'default filename')

    def test_save_method(self):
        if False:
            for i in range(10):
                print('nop')
        fn = next(filename_gen)
        with pytest.warns(UserWarning) as record:
            save_as_pdf_pages(p(), fn)
        assert_exist_and_clean(fn, 'save method')
        res = ('filename' in str(item.message).lower() for item in record)
        assert any(res)
        fn = next(filename_gen)
        with warnings.catch_warnings(record=True) as record:
            save_as_pdf_pages(p(), fn, verbose=False)
            assert_exist_and_clean(fn, 'save method')
            assert not record, 'Issued an unexpected warning'
        res = ('filename' in str(item.message).lower() for item in record)
        assert not any(res)

    def test_filename_plot_path(self):
        if False:
            while True:
                i = 10
        fn = next(filename_gen)
        with pytest.warns(PlotnineWarning):
            save_as_pdf_pages(p(), fn, path='.')
        assert_exist_and_clean(fn, 'fn, plot and path')

    @pytest.mark.skip('Results of this test can only be confirmed byinspecting the generated PDF.')
    def test_height_width(self):
        if False:
            i = 10
            return i + 15
        plots = []
        for (i, plot) in enumerate(p()):
            plots.append(plot + theme(figure_size=(8 + i, 6 + i)))
        fn = next(filename_gen)
        with pytest.warns(PlotnineWarning):
            save_as_pdf_pages(plots, fn)

class TestExceptions:

    def test_plot_exception(self):
        if False:
            print('Hello World!')
        fn = next(filename_gen)
        plots = list(p())
        plots[0] += aes(color='unknown')
        with pytest.raises(PlotnineError):
            save_as_pdf_pages(plots, fn, verbose=False)
        assert_exist_and_clean(fn, 'Plot exception')

def test_save_as_pdf_pages_closes_plots():
    if False:
        while True:
            i = 10
    assert plt.get_fignums() == [], 'There are unsaved test plots'
    fn = next(filename_gen)
    with pytest.warns(PlotnineWarning):
        save_as_pdf_pages(p(), fn)
    assert_exist_and_clean(fn, 'exist')
    assert plt.get_fignums() == [], 'ggplot.save did not close the plot'