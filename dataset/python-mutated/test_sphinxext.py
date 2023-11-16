"""Tests for tinypages build using sphinx extensions."""
import filecmp
import os
from pathlib import Path
import shutil
import sys
from matplotlib.testing import subprocess_run_for_testing
import pytest
pytest.importorskip('sphinx', minversion=None if sys.version_info < (3, 10) else '4.1.3')

def build_sphinx_html(source_dir, doctree_dir, html_dir, extra_args=None):
    if False:
        print('Hello World!')
    extra_args = [] if extra_args is None else extra_args
    cmd = [sys.executable, '-msphinx', '-W', '-b', 'html', '-d', str(doctree_dir), str(source_dir), str(html_dir), *extra_args]
    proc = subprocess_run_for_testing(cmd, capture_output=True, text=True, env={**os.environ, 'MPLBACKEND': ''})
    out = proc.stdout
    err = proc.stderr
    assert proc.returncode == 0, f'sphinx build failed with stdout:\n{out}\nstderr:\n{err}\n'
    if err:
        pytest.fail(f'sphinx build emitted the following warnings:\n{err}')
    assert html_dir.is_dir()

def test_tinypages(tmp_path):
    if False:
        i = 10
        return i + 15
    shutil.copytree(Path(__file__).parent / 'tinypages', tmp_path, dirs_exist_ok=True)
    html_dir = tmp_path / '_build' / 'html'
    img_dir = html_dir / '_images'
    doctree_dir = tmp_path / 'doctrees'
    cmd = [sys.executable, '-msphinx', '-W', '-b', 'html', '-d', str(doctree_dir), str(Path(__file__).parent / 'tinypages'), str(html_dir)]
    proc = subprocess_run_for_testing(cmd, capture_output=True, text=True, env={**os.environ, 'MPLBACKEND': '', 'GCOV_ERROR_FILE': os.devnull})
    out = proc.stdout
    err = proc.stderr
    build_sphinx_html(tmp_path, doctree_dir, html_dir)

    def plot_file(num):
        if False:
            for i in range(10):
                print('nop')
        return img_dir / f'some_plots-{num}.png'

    def plot_directive_file(num):
        if False:
            i = 10
            return i + 15
        return doctree_dir.parent / 'plot_directive' / f'some_plots-{num}.png'
    (range_10, range_6, range_4) = [plot_file(i) for i in range(1, 4)]
    assert filecmp.cmp(range_6, plot_file(5))
    assert filecmp.cmp(range_4, plot_file(7))
    assert filecmp.cmp(range_10, plot_file(11))
    assert filecmp.cmp(range_10, plot_file('12_00'))
    assert filecmp.cmp(range_6, plot_file('12_01'))
    assert filecmp.cmp(range_4, plot_file(13))
    html_contents = (html_dir / 'some_plots.html').read_bytes()
    assert b'# Only a comment' in html_contents
    assert filecmp.cmp(range_4, img_dir / 'range4.png')
    assert filecmp.cmp(range_6, img_dir / 'range6_range6.png')
    assert b'This is the caption for plot 15.' in html_contents
    assert b'Plot 17 uses the caption option.' in html_contents
    assert b'This is the caption for plot 18.' in html_contents
    assert b'plot-directive my-class my-other-class' in html_contents
    assert html_contents.count(b'This caption applies to both plots.') == 2
    assert filecmp.cmp(range_6, plot_file(17))
    assert filecmp.cmp(range_10, img_dir / 'range6_range10.png')
    contents = (tmp_path / 'included_plot_21.rst').read_bytes()
    contents = contents.replace(b'plt.plot(range(6))', b'plt.plot(range(4))')
    (tmp_path / 'included_plot_21.rst').write_bytes(contents)
    modification_times = [plot_directive_file(i).stat().st_mtime for i in (1, 2, 3, 5)]
    build_sphinx_html(tmp_path, doctree_dir, html_dir)
    assert filecmp.cmp(range_4, plot_file(17))
    assert plot_directive_file(1).stat().st_mtime == modification_times[0]
    assert plot_directive_file(2).stat().st_mtime == modification_times[1]
    assert plot_directive_file(3).stat().st_mtime == modification_times[2]
    assert filecmp.cmp(range_10, plot_file(1))
    assert filecmp.cmp(range_6, plot_file(2))
    assert filecmp.cmp(range_4, plot_file(3))
    assert plot_directive_file(5).stat().st_mtime > modification_times[3]
    assert filecmp.cmp(range_6, plot_file(5))

def test_plot_html_show_source_link(tmp_path):
    if False:
        print('Hello World!')
    parent = Path(__file__).parent
    shutil.copyfile(parent / 'tinypages/conf.py', tmp_path / 'conf.py')
    shutil.copytree(parent / 'tinypages/_static', tmp_path / '_static')
    doctree_dir = tmp_path / 'doctrees'
    (tmp_path / 'index.rst').write_text('\n.. plot::\n\n    plt.plot(range(2))\n')
    html_dir1 = tmp_path / '_build' / 'html1'
    build_sphinx_html(tmp_path, doctree_dir, html_dir1)
    assert len(list(html_dir1.glob('**/index-1.py'))) == 1
    html_dir2 = tmp_path / '_build' / 'html2'
    build_sphinx_html(tmp_path, doctree_dir, html_dir2, extra_args=['-D', 'plot_html_show_source_link=0'])
    assert len(list(html_dir2.glob('**/index-1.py'))) == 0

@pytest.mark.parametrize('plot_html_show_source_link', [0, 1])
def test_show_source_link_true(tmp_path, plot_html_show_source_link):
    if False:
        while True:
            i = 10
    parent = Path(__file__).parent
    shutil.copyfile(parent / 'tinypages/conf.py', tmp_path / 'conf.py')
    shutil.copytree(parent / 'tinypages/_static', tmp_path / '_static')
    doctree_dir = tmp_path / 'doctrees'
    (tmp_path / 'index.rst').write_text('\n.. plot::\n    :show-source-link: true\n\n    plt.plot(range(2))\n')
    html_dir = tmp_path / '_build' / 'html'
    build_sphinx_html(tmp_path, doctree_dir, html_dir, extra_args=['-D', f'plot_html_show_source_link={plot_html_show_source_link}'])
    assert len(list(html_dir.glob('**/index-1.py'))) == 1

@pytest.mark.parametrize('plot_html_show_source_link', [0, 1])
def test_show_source_link_false(tmp_path, plot_html_show_source_link):
    if False:
        while True:
            i = 10
    parent = Path(__file__).parent
    shutil.copyfile(parent / 'tinypages/conf.py', tmp_path / 'conf.py')
    shutil.copytree(parent / 'tinypages/_static', tmp_path / '_static')
    doctree_dir = tmp_path / 'doctrees'
    (tmp_path / 'index.rst').write_text('\n.. plot::\n    :show-source-link: false\n\n    plt.plot(range(2))\n')
    html_dir = tmp_path / '_build' / 'html'
    build_sphinx_html(tmp_path, doctree_dir, html_dir, extra_args=['-D', f'plot_html_show_source_link={plot_html_show_source_link}'])
    assert len(list(html_dir.glob('**/index-1.py'))) == 0

def test_srcset_version(tmp_path):
    if False:
        return 10
    shutil.copytree(Path(__file__).parent / 'tinypages', tmp_path, dirs_exist_ok=True)
    html_dir = tmp_path / '_build' / 'html'
    img_dir = html_dir / '_images'
    doctree_dir = tmp_path / 'doctrees'
    build_sphinx_html(tmp_path, doctree_dir, html_dir, extra_args=['-D', 'plot_srcset=2x'])

    def plot_file(num, suff=''):
        if False:
            i = 10
            return i + 15
        return img_dir / f'some_plots-{num}{suff}.png'
    for ind in [1, 2, 3, 5, 7, 11, 13, 15, 17]:
        assert plot_file(ind).exists()
        assert plot_file(ind, suff='.2x').exists()
    assert (img_dir / 'nestedpage-index-1.png').exists()
    assert (img_dir / 'nestedpage-index-1.2x.png').exists()
    assert (img_dir / 'nestedpage-index-2.png').exists()
    assert (img_dir / 'nestedpage-index-2.2x.png').exists()
    assert (img_dir / 'nestedpage2-index-1.png').exists()
    assert (img_dir / 'nestedpage2-index-1.2x.png').exists()
    assert (img_dir / 'nestedpage2-index-2.png').exists()
    assert (img_dir / 'nestedpage2-index-2.2x.png').exists()
    assert 'srcset="_images/some_plots-1.png, _images/some_plots-1.2x.png 2.00x"' in (html_dir / 'some_plots.html').read_text(encoding='utf-8')
    st = 'srcset="../_images/nestedpage-index-1.png, ../_images/nestedpage-index-1.2x.png 2.00x"'
    assert st in (html_dir / 'nestedpage/index.html').read_text(encoding='utf-8')
    st = 'srcset="../_images/nestedpage2-index-2.png, ../_images/nestedpage2-index-2.2x.png 2.00x"'
    assert st in (html_dir / 'nestedpage2/index.html').read_text(encoding='utf-8')