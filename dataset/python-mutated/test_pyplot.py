import difflib
import numpy as np
import sys
from pathlib import Path
import pytest
import matplotlib as mpl
from matplotlib.testing import subprocess_run_for_testing
from matplotlib import pyplot as plt

def test_pyplot_up_to_date(tmpdir):
    if False:
        print('Hello World!')
    pytest.importorskip('black')
    gen_script = Path(mpl.__file__).parents[2] / 'tools/boilerplate.py'
    if not gen_script.exists():
        pytest.skip('boilerplate.py not found')
    orig_contents = Path(plt.__file__).read_text()
    plt_file = tmpdir.join('pyplot.py')
    plt_file.write_text(orig_contents, 'utf-8')
    subprocess_run_for_testing([sys.executable, str(gen_script), str(plt_file)], check=True)
    new_contents = plt_file.read_text('utf-8')
    if orig_contents != new_contents:
        diff_msg = '\n'.join(difflib.unified_diff(orig_contents.split('\n'), new_contents.split('\n'), fromfile='found pyplot.py', tofile='expected pyplot.py', n=0, lineterm=''))
        pytest.fail("pyplot.py is not up-to-date. Please run 'python tools/boilerplate.py' to update pyplot.py. This needs to be done from an environment where your current working copy is installed (e.g. 'pip install -e'd). Here is a diff of unexpected differences:\n%s" % diff_msg)

def test_copy_docstring_and_deprecators(recwarn):
    if False:
        i = 10
        return i + 15

    @mpl._api.rename_parameter('(version)', 'old', 'new')
    @mpl._api.make_keyword_only('(version)', 'kwo')
    def func(new, kwo=None):
        if False:
            for i in range(10):
                print('nop')
        pass

    @plt._copy_docstring_and_deprecators(func)
    def wrapper_func(new, kwo=None):
        if False:
            return 10
        pass
    wrapper_func(None)
    wrapper_func(new=None)
    wrapper_func(None, kwo=None)
    wrapper_func(new=None, kwo=None)
    assert not recwarn
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        wrapper_func(old=None)
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        wrapper_func(None, None)

def test_pyplot_box():
    if False:
        for i in range(10):
            print('nop')
    (fig, ax) = plt.subplots()
    plt.box(False)
    assert not ax.get_frame_on()
    plt.box(True)
    assert ax.get_frame_on()
    plt.box()
    assert not ax.get_frame_on()
    plt.box()
    assert ax.get_frame_on()

def test_stackplot_smoke():
    if False:
        return 10
    plt.stackplot([1, 2, 3], [1, 2, 3])

def test_nrows_error():
    if False:
        i = 10
        return i + 15
    with pytest.raises(TypeError):
        plt.subplot(nrows=1)
    with pytest.raises(TypeError):
        plt.subplot(ncols=1)

def test_ioff():
    if False:
        i = 10
        return i + 15
    plt.ion()
    assert mpl.is_interactive()
    with plt.ioff():
        assert not mpl.is_interactive()
    assert mpl.is_interactive()
    plt.ioff()
    assert not mpl.is_interactive()
    with plt.ioff():
        assert not mpl.is_interactive()
    assert not mpl.is_interactive()

def test_ion():
    if False:
        for i in range(10):
            print('nop')
    plt.ioff()
    assert not mpl.is_interactive()
    with plt.ion():
        assert mpl.is_interactive()
    assert not mpl.is_interactive()
    plt.ion()
    assert mpl.is_interactive()
    with plt.ion():
        assert mpl.is_interactive()
    assert mpl.is_interactive()

def test_nested_ion_ioff():
    if False:
        return 10
    plt.ion()
    with plt.ioff():
        assert not mpl.is_interactive()
        with plt.ion():
            assert mpl.is_interactive()
        assert not mpl.is_interactive()
    assert mpl.is_interactive()
    with plt.ioff():
        with plt.ioff():
            assert not mpl.is_interactive()
    assert mpl.is_interactive()
    with plt.ion():
        plt.ioff()
    assert mpl.is_interactive()
    plt.ioff()
    with plt.ion():
        assert mpl.is_interactive()
        with plt.ioff():
            assert not mpl.is_interactive()
        assert mpl.is_interactive()
    assert not mpl.is_interactive()
    with plt.ion():
        with plt.ion():
            assert mpl.is_interactive()
    assert not mpl.is_interactive()
    with plt.ioff():
        plt.ion()
    assert not mpl.is_interactive()

def test_close():
    if False:
        for i in range(10):
            print('nop')
    try:
        plt.close(1.1)
    except TypeError as e:
        assert str(e) == "close() argument must be a Figure, an int, a string, or None, not <class 'float'>"

def test_subplot_reuse():
    if False:
        print('Hello World!')
    ax1 = plt.subplot(121)
    assert ax1 is plt.gca()
    ax2 = plt.subplot(122)
    assert ax2 is plt.gca()
    ax3 = plt.subplot(121)
    assert ax1 is plt.gca()
    assert ax1 is ax3

def test_axes_kwargs():
    if False:
        return 10
    plt.figure()
    ax = plt.axes()
    ax1 = plt.axes()
    assert ax is not None
    assert ax1 is not ax
    plt.close()
    plt.figure()
    ax = plt.axes(projection='polar')
    ax1 = plt.axes(projection='polar')
    assert ax is not None
    assert ax1 is not ax
    plt.close()
    plt.figure()
    ax = plt.axes(projection='polar')
    ax1 = plt.axes()
    assert ax is not None
    assert ax1.name == 'rectilinear'
    assert ax1 is not ax
    plt.close()

def test_subplot_replace_projection():
    if False:
        for i in range(10):
            print('nop')
    fig = plt.figure()
    ax = plt.subplot(1, 2, 1)
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    ax3 = plt.subplot(1, 2, 1, projection='polar')
    ax4 = plt.subplot(1, 2, 1, projection='polar')
    assert ax is not None
    assert ax1 is ax
    assert ax2 is not ax
    assert ax3 is not ax
    assert ax3 is ax4
    assert ax in fig.axes
    assert ax2 in fig.axes
    assert ax3 in fig.axes
    assert ax.name == 'rectilinear'
    assert ax2.name == 'rectilinear'
    assert ax3.name == 'polar'

def test_subplot_kwarg_collision():
    if False:
        for i in range(10):
            print('nop')
    ax1 = plt.subplot(projection='polar', theta_offset=0)
    ax2 = plt.subplot(projection='polar', theta_offset=0)
    assert ax1 is ax2
    ax1.remove()
    ax3 = plt.subplot(projection='polar', theta_offset=1)
    assert ax1 is not ax3
    assert ax1 not in plt.gcf().axes

def test_gca():
    if False:
        i = 10
        return i + 15
    plt.figure()
    ax = plt.gca()
    ax1 = plt.gca()
    assert ax is not None
    assert ax1 is ax
    plt.close()

def test_subplot_projection_reuse():
    if False:
        print('Hello World!')
    ax1 = plt.subplot(111)
    assert ax1 is plt.gca()
    assert ax1 is plt.subplot(111)
    ax1.remove()
    ax2 = plt.subplot(111, projection='polar')
    assert ax2 is plt.gca()
    assert ax1 not in plt.gcf().axes
    assert ax2 is plt.subplot(111)
    ax2.remove()
    ax3 = plt.subplot(111, projection='rectilinear')
    assert ax3 is plt.gca()
    assert ax3 is not ax2
    assert ax2 not in plt.gcf().axes

def test_subplot_polar_normalization():
    if False:
        for i in range(10):
            print('nop')
    ax1 = plt.subplot(111, projection='polar')
    ax2 = plt.subplot(111, polar=True)
    ax3 = plt.subplot(111, polar=True, projection='polar')
    assert ax1 is ax2
    assert ax1 is ax3
    with pytest.raises(ValueError, match="polar=True, yet projection='3d'"):
        ax2 = plt.subplot(111, polar=True, projection='3d')

def test_subplot_change_projection():
    if False:
        while True:
            i = 10
    created_axes = set()
    ax = plt.subplot()
    created_axes.add(ax)
    projections = ('aitoff', 'hammer', 'lambert', 'mollweide', 'polar', 'rectilinear', '3d')
    for proj in projections:
        ax.remove()
        ax = plt.subplot(projection=proj)
        assert ax is plt.subplot()
        assert ax.name == proj
        created_axes.add(ax)
    assert len(created_axes) == 1 + len(projections)

def test_polar_second_call():
    if False:
        while True:
            i = 10
    (ln1,) = plt.polar(0.0, 1.0, 'ro')
    assert isinstance(ln1, mpl.lines.Line2D)
    (ln2,) = plt.polar(1.57, 0.5, 'bo')
    assert isinstance(ln2, mpl.lines.Line2D)
    assert ln1.axes is ln2.axes

def test_fallback_position():
    if False:
        while True:
            i = 10
    axref = plt.axes([0.2, 0.2, 0.5, 0.5])
    axtest = plt.axes(position=[0.2, 0.2, 0.5, 0.5])
    np.testing.assert_allclose(axtest.bbox.get_points(), axref.bbox.get_points())
    axref = plt.axes([0.2, 0.2, 0.5, 0.5])
    axtest = plt.axes([0.2, 0.2, 0.5, 0.5], position=[0.1, 0.1, 0.8, 0.8])
    np.testing.assert_allclose(axtest.bbox.get_points(), axref.bbox.get_points())

def test_set_current_figure_via_subfigure():
    if False:
        print('Hello World!')
    fig1 = plt.figure()
    subfigs = fig1.subfigures(2)
    plt.figure()
    assert plt.gcf() != fig1
    current = plt.figure(subfigs[1])
    assert plt.gcf() == fig1
    assert current == fig1

def test_set_current_axes_on_subfigure():
    if False:
        i = 10
        return i + 15
    fig = plt.figure()
    subfigs = fig.subfigures(2)
    ax = subfigs[0].subplots(1, squeeze=True)
    subfigs[1].subplots(1, squeeze=True)
    assert plt.gca() != ax
    plt.sca(ax)
    assert plt.gca() == ax

def test_pylab_integration():
    if False:
        for i in range(10):
            print('nop')
    IPython = pytest.importorskip('IPython')
    mpl.testing.subprocess_run_helper(IPython.start_ipython, '--pylab', '-c', ';'.join(('import matplotlib.pyplot as plt', 'assert plt._REPL_DISPLAYHOOK == plt._ReplDisplayHook.IPYTHON')), timeout=60)

def test_doc_pyplot_summary():
    if False:
        while True:
            i = 10
    'Test that pyplot_summary lists all the plot functions.'
    pyplot_docs = Path(__file__).parent / '../../../doc/api/pyplot_summary.rst'
    if not pyplot_docs.exists():
        pytest.skip('Documentation sources not available')

    def extract_documented_functions(lines):
        if False:
            return 10
        '\n        Return a list of all the functions that are mentioned in the\n        autosummary blocks contained in *lines*.\n\n        An autosummary block looks like this::\n\n            .. autosummary::\n               :toctree: _as_gen\n               :template: autosummary.rst\n               :nosignatures:\n\n               plot\n               plot_date\n\n        '
        functions = []
        in_autosummary = False
        for line in lines:
            if not in_autosummary:
                if line.startswith('.. autosummary::'):
                    in_autosummary = True
            else:
                if not line or line.startswith('   :'):
                    continue
                if not line[0].isspace():
                    in_autosummary = False
                    continue
                functions.append(line.strip())
        return functions
    lines = pyplot_docs.read_text().split('\n')
    doc_functions = set(extract_documented_functions(lines))
    plot_commands = set(plt._get_pyplot_commands())
    missing = plot_commands.difference(doc_functions)
    if missing:
        raise AssertionError(f'The following pyplot functions are not listed in the documentation. Please add them to doc/api/pyplot_summary.rst: {missing!r}')
    extra = doc_functions.difference(plot_commands)
    if extra:
        raise AssertionError(f'The following functions are listed in the pyplot documentation, but they do not exist in pyplot. Please remove them from doc/api/pyplot_summary.rst: {extra!r}')

def test_minor_ticks():
    if False:
        for i in range(10):
            print('nop')
    plt.figure()
    plt.plot(np.arange(1, 10))
    (tick_pos, tick_labels) = plt.xticks(minor=True)
    assert np.all(tick_labels == np.array([], dtype=np.float64))
    assert tick_labels == []
    plt.yticks(ticks=[3.5, 6.5], labels=['a', 'b'], minor=True)
    ax = plt.gca()
    tick_pos = ax.get_yticks(minor=True)
    tick_labels = ax.get_yticklabels(minor=True)
    assert np.all(tick_pos == np.array([3.5, 6.5]))
    assert [l.get_text() for l in tick_labels] == ['a', 'b']

def test_switch_backend_no_close():
    if False:
        for i in range(10):
            print('nop')
    plt.switch_backend('agg')
    fig = plt.figure()
    fig = plt.figure()
    assert len(plt.get_fignums()) == 2
    plt.switch_backend('agg')
    assert len(plt.get_fignums()) == 2
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        plt.switch_backend('svg')
    assert len(plt.get_fignums()) == 0

def figure_hook_example(figure):
    if False:
        i = 10
        return i + 15
    figure._test_was_here = True

def test_figure_hook():
    if False:
        print('Hello World!')
    test_rc = {'figure.hooks': ['matplotlib.tests.test_pyplot:figure_hook_example']}
    with mpl.rc_context(test_rc):
        fig = plt.figure()
    assert fig._test_was_here