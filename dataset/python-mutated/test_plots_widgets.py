"""
Tests for the widgets used in the Plots plugin.
"""
import os.path as osp
import datetime
from unittest.mock import Mock
import pytest
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
from qtpy.QtWidgets import QApplication, QStyle
from qtpy.QtGui import QPixmap
from qtpy.QtCore import Qt
from spyder.plugins.plots.widgets.figurebrowser import FigureBrowser, FigureThumbnail
from spyder.plugins.plots.widgets.figurebrowser import get_unique_figname

@pytest.fixture
def figbrowser(qtbot):
    if False:
        i = 10
        return i + 15
    'An empty figure browser widget fixture.'
    figbrowser = FigureBrowser()
    figbrowser.set_shellwidget(Mock())
    options = {'mute_inline_plotting': True, 'show_plot_outline': False, 'auto_fit_plotting': False}
    figbrowser.setup(options)
    qtbot.addWidget(figbrowser)
    figbrowser.show()
    figbrowser.setMinimumSize(700, 500)
    return figbrowser

def create_figure(figname):
    if False:
        return 10
    'Create a matplotlib figure, save it to disk and return its data.'
    fig = Figure()
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
    fig.set_size_inches(6, 4)
    ax.plot(np.random.rand(10), '.', color='red')
    fig.savefig(figname)
    with open(figname, 'rb') as img:
        fig = img.read()
    return fig

def add_figures_to_browser(figbrowser, nfig, tmpdir, fmt='image/png'):
    if False:
        i = 10
        return i + 15
    '\n    Create and add bitmap figures to the figure browser. Also return a list\n    of the created figures data.\n    '
    fext = '.svg' if fmt == 'image/svg+xml' else '.png'
    figs = []
    for i in range(nfig):
        figname = osp.join(str(tmpdir), 'mplfig' + str(i) + fext)
        figs.append(create_figure(figname))
        figbrowser.add_figure(figs[-1], fmt)
    assert len(figbrowser.thumbnails_sb._thumbnails) == nfig
    assert figbrowser.thumbnails_sb.get_current_index() == nfig - 1
    assert figbrowser.thumbnails_sb.current_thumbnail.canvas.fig == figs[-1]
    assert figbrowser.figviewer.figcanvas.fig == figs[-1]
    return figs

def png_to_qimage(png):
    if False:
        while True:
            i = 10
    'Return a QImage from the raw data of a png image.'
    qpix = QPixmap()
    qpix.loadFromData(png, 'image/png'.upper())
    return qpix.toImage()

@pytest.mark.order(1)
@pytest.mark.parametrize('fmt, fext', [('image/png', '.png'), ('image/svg+xml', '.svg')])
def test_add_figures(figbrowser, tmpdir, fmt, fext):
    if False:
        while True:
            i = 10
    '\n    Test that the figure browser widget display correctly new figures in\n    its viewer and thumbnails scrollbar.\n    '
    assert len(figbrowser.thumbnails_sb._thumbnails) == 0
    assert figbrowser.thumbnails_sb.current_thumbnail is None
    assert figbrowser.figviewer.figcanvas.fig is None
    for i in range(3):
        figname = osp.join(str(tmpdir), 'mplfig' + str(i) + fext)
        fig = create_figure(figname)
        figbrowser.add_figure(fig, fmt)
        assert len(figbrowser.thumbnails_sb._thumbnails) == i + 1
        assert figbrowser.thumbnails_sb.get_current_index() == i
        assert figbrowser.thumbnails_sb.current_thumbnail.canvas.fig == fig
        assert figbrowser.figviewer.figcanvas.fig == fig

@pytest.mark.parametrize('fmt, fext', [('image/png', '.png'), ('image/svg+xml', '.svg'), ('image/svg+xml', '.png')])
def test_save_figure_to_file(figbrowser, tmpdir, mocker, fmt, fext):
    if False:
        return 10
    '\n    Test saving png and svg figures to file with the figure browser.\n    '
    fig = add_figures_to_browser(figbrowser, 1, tmpdir, fmt)[0]
    expected_qpix = QPixmap()
    expected_qpix.loadFromData(fig, fmt.upper())
    saved_figname = osp.join(str(tmpdir), 'spyfig' + fext)
    mocker.patch('spyder.plugins.plots.widgets.figurebrowser.getsavefilename', return_value=(saved_figname, fext))
    figbrowser.save_figure()
    saved_qpix = QPixmap()
    saved_qpix.load(saved_figname)
    assert osp.exists(saved_figname)
    assert expected_qpix.toImage() == saved_qpix.toImage()

@pytest.mark.parametrize('fmt', ['image/png', 'image/svg+xml'])
def test_save_all_figures(figbrowser, tmpdir, mocker, fmt):
    if False:
        i = 10
        return i + 15
    '\n    Test saving all figures contained in the thumbnail scrollbar in batch\n    into a single directory.\n    '
    figs = add_figures_to_browser(figbrowser, 3, tmpdir, fmt)
    mocker.patch('spyder.plugins.plots.widgets.figurebrowser.getexistingdirectory', return_value=None)
    fignames = figbrowser.save_all_figures()
    assert fignames is None
    mocker.patch('spyder.plugins.plots.widgets.figurebrowser.getexistingdirectory', return_value=str(tmpdir.mkdir('all_saved_figures')))
    fignames = figbrowser.save_all_figures()
    assert len(fignames) == len(figs)
    for (fig, figname) in zip(figs, fignames):
        expected_qpix = QPixmap()
        expected_qpix.loadFromData(fig, fmt.upper())
        saved_qpix = QPixmap()
        saved_qpix.load(figname)
        assert osp.exists(figname)
        assert expected_qpix.toImage() == saved_qpix.toImage()

def test_get_unique_figname(tmpdir):
    if False:
        return 10
    '\n    Test that the unique fig names work when saving only one and when\n    saving multiple figures.\n    '
    fext = '.png'
    figname_root = 'Figure ' + datetime.datetime.now().strftime('%Y-%m-%d %H%M%S')
    figname = get_unique_figname(tmpdir, figname_root, fext)
    expected = osp.join(tmpdir, '{}{}'.format(figname_root, fext))
    assert figname == expected
    figname_root = 'Figure ' + datetime.datetime.now().strftime('%Y-%m-%d %H%M%S')
    for i in range(5):
        figname = get_unique_figname(tmpdir, figname_root, fext, start_at_zero=True)
        with open(figname, 'w') as _:
            pass
        expected = osp.join(tmpdir, '{} ({}){}'.format(figname_root, i, fext))
        assert figname == expected

@pytest.mark.parametrize('fmt', ['image/png', 'image/svg+xml'])
def test_close_current_figure(figbrowser, tmpdir, fmt):
    if False:
        while True:
            i = 10
    '\n    Test that clearing the current figure works as expected.\n    '
    figs = add_figures_to_browser(figbrowser, 2, tmpdir, fmt)
    figbrowser.close_figure()
    assert len(figbrowser.thumbnails_sb._thumbnails) == 1
    assert figbrowser.thumbnails_sb.get_current_index() == 0
    assert figbrowser.thumbnails_sb.current_thumbnail.canvas.fig == figs[0]
    assert figbrowser.figviewer.figcanvas.fig == figs[0]
    figbrowser.close_figure()
    assert len(figbrowser.thumbnails_sb._thumbnails) == 0
    assert figbrowser.thumbnails_sb.get_current_index() == -1
    assert figbrowser.thumbnails_sb.current_thumbnail is None
    assert figbrowser.figviewer.figcanvas.fig is None

@pytest.mark.parametrize('fmt', ['image/png', 'image/svg+xml'])
def test_close_all_figures(figbrowser, tmpdir, fmt):
    if False:
        while True:
            i = 10
    '\n    Test that clearing all figures displayed in the thumbnails scrollbar\n    works as expected.\n    '
    add_figures_to_browser(figbrowser, 3, tmpdir, fmt)
    figbrowser.close_all_figures()
    assert len(figbrowser.thumbnails_sb._thumbnails) == 0
    assert figbrowser.thumbnails_sb.get_current_index() == -1
    assert figbrowser.thumbnails_sb.current_thumbnail is None
    assert figbrowser.figviewer.figcanvas.fig is None
    assert len(figbrowser.thumbnails_sb.findChildren(FigureThumbnail)) == 0

@pytest.mark.parametrize('fmt', ['image/png', 'image/svg+xml'])
def test_close_one_thumbnail(qtbot, figbrowser, tmpdir, fmt):
    if False:
        while True:
            i = 10
    '\n    Test the thumbnail is removed from the GUI.\n    '
    add_figures_to_browser(figbrowser, 2, tmpdir, fmt)
    assert len(figbrowser.thumbnails_sb.findChildren(FigureThumbnail)) == 2
    figures = figbrowser.thumbnails_sb.findChildren(FigureThumbnail)
    figbrowser.thumbnails_sb.remove_thumbnail(figures[0])
    qtbot.wait(200)
    assert len(figbrowser.thumbnails_sb.findChildren(FigureThumbnail)) == 1

@pytest.mark.parametrize('fmt', ['image/png', 'image/svg+xml'])
def test_go_prev_next_thumbnail(figbrowser, tmpdir, fmt):
    if False:
        return 10
    '\n    Test go to previous and next thumbnail actions.\n    '
    figs = add_figures_to_browser(figbrowser, 3, tmpdir, fmt)
    figbrowser.go_next_thumbnail()
    assert figbrowser.thumbnails_sb.get_current_index() == 0
    assert figbrowser.thumbnails_sb.current_thumbnail.canvas.fig == figs[0]
    assert figbrowser.figviewer.figcanvas.fig == figs[0]
    figbrowser.go_previous_thumbnail()
    assert figbrowser.thumbnails_sb.get_current_index() == 2
    assert figbrowser.thumbnails_sb.current_thumbnail.canvas.fig == figs[2]
    assert figbrowser.figviewer.figcanvas.fig == figs[2]
    figbrowser.go_previous_thumbnail()
    assert figbrowser.thumbnails_sb.get_current_index() == 1
    assert figbrowser.thumbnails_sb.current_thumbnail.canvas.fig == figs[1]
    assert figbrowser.figviewer.figcanvas.fig == figs[1]

def test_scroll_to_item(figbrowser, tmpdir, qtbot):
    if False:
        return 10
    'Test scroll to the item of ThumbnailScrollBar.'
    nfig = 10
    add_figures_to_browser(figbrowser, nfig, tmpdir, 'image/png')
    figbrowser.setFixedSize(500, 500)
    for __ in range(nfig // 2):
        figbrowser.go_next_thumbnail()
        qtbot.wait(500)
    scene = figbrowser.thumbnails_sb.scene
    spacing = scene.verticalSpacing()
    height = scene.itemAt(0).sizeHint().height()
    height_view = figbrowser.thumbnails_sb.scrollarea.viewport().height()
    expected = spacing * (nfig // 2) + height * (nfig // 2 - 1) - (height_view - height) // 2
    vsb = figbrowser.thumbnails_sb.scrollarea.verticalScrollBar()
    assert vsb.value() == expected

def test_scroll_down_to_newest_plot(figbrowser, tmpdir, qtbot):
    if False:
        i = 10
        return i + 15
    '\n    Test that the ThumbnailScrollBar is scrolled to the newest plot after\n    it is added to it.\n\n    Test that covers spyder-ide/spyder#10914.\n    '
    figbrowser.setFixedSize(500, 500)
    nfig = 8
    for i in range(8):
        newfig = create_figure(osp.join(str(tmpdir), 'new_mplfig{}.png'.format(i)))
        figbrowser.add_figure(newfig, 'image/png')
        qtbot.wait(500)
    height_view = figbrowser.thumbnails_sb.scrollarea.viewport().height()
    scene = figbrowser.thumbnails_sb.scene
    spacing = scene.verticalSpacing()
    height = scene.itemAt(0).sizeHint().height()
    expected = spacing * (nfig - 1) + height * nfig - height_view
    vsb = figbrowser.thumbnails_sb.scrollarea.verticalScrollBar()
    assert vsb.value() == expected

@pytest.mark.parametrize('fmt', ['image/png', 'image/svg+xml'])
def test_mouse_clicking_thumbnails(figbrowser, tmpdir, qtbot, fmt):
    if False:
        i = 10
        return i + 15
    '\n    Test mouse clicking on thumbnails.\n    '
    figs = add_figures_to_browser(figbrowser, 3, tmpdir, fmt)
    for i in [1, 0, 2]:
        qtbot.mouseClick(figbrowser.thumbnails_sb._thumbnails[i].canvas, Qt.LeftButton)
        assert figbrowser.thumbnails_sb.get_current_index() == i
        assert figbrowser.thumbnails_sb.current_thumbnail.canvas.fig == figs[i]
        assert figbrowser.figviewer.figcanvas.fig == figs[i]

@pytest.mark.parametrize('fmt', ['image/png', 'image/svg+xml'])
def test_save_thumbnails(figbrowser, tmpdir, qtbot, mocker, fmt):
    if False:
        while True:
            i = 10
    '\n    Test saving figures by clicking on the thumbnail icon.\n    '
    figs = add_figures_to_browser(figbrowser, 3, tmpdir, fmt)
    fext = '.svg' if fmt == 'image/svg+xml' else '.png'
    figname = osp.join(str(tmpdir), 'figname' + fext)
    mocker.patch('spyder.plugins.plots.widgets.figurebrowser.getsavefilename', return_value=(figname, fext))
    figbrowser.thumbnails_sb.set_current_index(1)
    figbrowser.save_figure()
    expected_qpix = QPixmap()
    expected_qpix.loadFromData(figs[1], fmt.upper())
    saved_qpix = QPixmap()
    saved_qpix.load(figname)
    assert osp.exists(figname)
    assert expected_qpix.toImage() == saved_qpix.toImage()

@pytest.mark.parametrize('fmt', ['image/png', 'image/svg+xml'])
def test_close_thumbnails(figbrowser, tmpdir, qtbot, mocker, fmt):
    if False:
        while True:
            i = 10
    '\n    Test closing figures by clicking on the thumbnail icon.\n    '
    figs = add_figures_to_browser(figbrowser, 3, tmpdir, fmt)
    figbrowser.thumbnails_sb.set_current_index(1)
    figbrowser.close_figure()
    del figs[1]
    assert len(figbrowser.thumbnails_sb._thumbnails) == len(figs)
    assert figbrowser.thumbnails_sb._thumbnails[0].canvas.fig == figs[0]
    assert figbrowser.thumbnails_sb._thumbnails[1].canvas.fig == figs[1]

def test_copy_png_to_clipboard(figbrowser, tmpdir):
    if False:
        while True:
            i = 10
    '\n    Test copying png figures to the clipboard.\n    '
    figs = add_figures_to_browser(figbrowser, 3, tmpdir, 'image/png')
    clipboard = QApplication.clipboard()
    figbrowser.copy_figure()
    assert clipboard.image() == png_to_qimage(figs[-1])
    figbrowser.go_next_thumbnail()
    figbrowser.copy_figure()
    assert clipboard.image() == png_to_qimage(figs[0])

def test_copy_svg_to_clipboard(figbrowser, tmpdir):
    if False:
        print('Hello World!')
    '\n    Test copying svg figures to the clipboard.\n    '
    figs = add_figures_to_browser(figbrowser, 3, tmpdir, 'image/svg+xml')
    clipboard = QApplication.clipboard()
    figbrowser.copy_figure()
    assert clipboard.mimeData().data('image/svg+xml') == figs[-1]
    figbrowser.go_next_thumbnail()
    figbrowser.copy_figure()
    assert clipboard.mimeData().data('image/svg+xml') == figs[0]

@pytest.mark.parametrize('fmt', ['image/png', 'image/svg+xml'])
def test_zoom_figure_viewer(figbrowser, tmpdir, fmt):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test zooming in and out the figure diplayed in the figure viewer.\n    '
    fig = add_figures_to_browser(figbrowser, 1, tmpdir, fmt)[0]
    figcanvas = figbrowser.figviewer.figcanvas
    figbrowser.change_auto_fit_plotting(False)
    qpix = QPixmap()
    qpix.loadFromData(fig, fmt.upper())
    (fwidth, fheight) = (qpix.width(), qpix.height())
    assert figbrowser.zoom_disp_value == 100
    assert figcanvas.width() == fwidth
    assert figcanvas.height() == fheight
    scaling_factor = 0
    scaling_step = figbrowser.figviewer._scalestep
    for zoom_step in [1, 1, -1, -1, -1]:
        if zoom_step == 1:
            figbrowser.zoom_in()
        elif zoom_step == -1:
            figbrowser.zoom_out()
        scaling_factor += zoom_step
        scale = scaling_step ** scaling_factor
        assert figbrowser.zoom_disp_value == np.round(int(fwidth * scale) / fwidth * 100)
        assert figcanvas.width() == int(fwidth * scale)
        assert figcanvas.height() == int(fheight * scale)

@pytest.mark.parametrize('fmt', ['image/png', 'image/svg+xml'])
def test_autofit_figure_viewer(figbrowser, tmpdir, fmt):
    if False:
        return 10
    '\n    Test figure diplayed when `Fit plots to window` is True.\n    '
    fig = add_figures_to_browser(figbrowser, 1, tmpdir, fmt)[0]
    figviewer = figbrowser.figviewer
    figcanvas = figviewer.figcanvas
    qpix = QPixmap()
    qpix.loadFromData(fig, fmt.upper())
    (fwidth, fheight) = (qpix.width(), qpix.height())
    figbrowser.change_auto_fit_plotting(True)
    size = figviewer.size()
    style = figviewer.style()
    width = size.width() - style.pixelMetric(QStyle.PM_LayoutLeftMargin) - style.pixelMetric(QStyle.PM_LayoutRightMargin)
    height = size.height() - style.pixelMetric(QStyle.PM_LayoutTopMargin) - style.pixelMetric(QStyle.PM_LayoutBottomMargin)
    if fwidth / fheight > width / height:
        new_width = int(width)
        new_height = int(width / fwidth * fheight)
    else:
        new_height = int(height)
        new_width = int(height / fheight * fwidth)
    assert figcanvas.width() == new_width
    assert figcanvas.height() == new_height
    assert figbrowser.zoom_disp_value == round(figcanvas.width() / fwidth * 100)
if __name__ == '__main__':
    pytest.main()