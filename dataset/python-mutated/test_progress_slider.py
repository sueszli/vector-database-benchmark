import pytest
from PyQt5.QtWidgets import QAbstractSlider
from feeluown.gui.widgets.progress_slider import ProgressSlider

@pytest.fixture
def mock_update_player(mocker):
    if False:
        for i in range(10):
            print('nop')
    return mocker.patch.object(ProgressSlider, 'maybe_update_player_position')

@pytest.fixture
def slider(qtbot, app_mock):
    if False:
        print('Hello World!')
    slider = ProgressSlider(app_mock)
    app_mock.player.current_media = object()
    app_mock.player.position = 0
    qtbot.addWidget(slider)
    return slider

def test_basics(qtbot, app_mock):
    if False:
        for i in range(10):
            print('nop')
    slider = ProgressSlider(app_mock)
    qtbot.addWidget(slider)

def test_action_is_triggered(slider, mock_update_player):
    if False:
        while True:
            i = 10
    slider.triggerAction(QAbstractSlider.SliderPageStepAdd)
    assert mock_update_player.called

def test_maybe_update_player_position(slider):
    if False:
        return 10
    slider.maybe_update_player_position(10)
    assert slider._app.player.position == 10
    assert slider._app.player.resume.called

def test_update_total(slider):
    if False:
        for i in range(10):
            print('nop')
    slider.update_total(10)
    assert slider.maximum() == 10

def test_drag_slider(slider, mock_update_player):
    if False:
        return 10
    slider.setSliderDown(True)
    slider.setSliderDown(False)
    assert mock_update_player.called

def test_media_changed_during_dragging(qtbot, slider, mock_update_player):
    if False:
        i = 10
        return i + 15
    slider.setSliderDown(True)
    slider._dragging_ctx.is_media_changed = True
    slider.setSliderDown(False)
    assert not mock_update_player.called

def test_when_player_has_no_media(slider):
    if False:
        while True:
            i = 10
    slider._app.player.current_media = None
    slider.triggerAction(QAbstractSlider.SliderPageStepAdd)
    assert not slider._app.player.resume.called