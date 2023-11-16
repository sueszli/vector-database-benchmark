from typing import Optional
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSlider, QAbstractSlider
from feeluown.player import State

class DraggingContext:

    def __init__(self):
        if False:
            return 10
        self.is_media_changed = False

class ProgressSlider(QSlider):

    def __init__(self, app, parent=None):
        if False:
            i = 10
            return i + 15
        super().__init__(parent)
        self._app = app
        self._dragging_ctx: Optional[DraggingContext] = None
        self.setToolTip('拖动调节进度')
        self.setRange(0, 0)
        self.setOrientation(Qt.Horizontal)
        self.sliderPressed.connect(self.on_pressed)
        self.sliderReleased.connect(self.on_released)
        self.actionTriggered.connect(self.on_action_triggered)
        self._app.player.duration_changed.connect(self.update_total, aioqueue=True)
        self._app.player_pos_per300ms.changed.connect(self.update_progress)
        self._app.player.media_changed.connect(self.on_media_changed)

    def update_total(self, s):
        if False:
            print('Hello World!')
        s = s or 0
        self.setRange(0, int(s))

    def update_progress(self, s):
        if False:
            i = 10
            return i + 15
        if not self.is_dragging:
            s = s or 0
            self.setValue(int(s))

    @property
    def is_dragging(self):
        if False:
            i = 10
            return i + 15
        return self._dragging_ctx is not None

    def on_pressed(self):
        if False:
            print('Hello World!')
        self._dragging_ctx = DraggingContext()
        if self._app.player.state is State.playing:
            self._app.player.pause()

    def on_released(self):
        if False:
            for i in range(10):
                print('nop')
        assert self._dragging_ctx is not None
        if not self._dragging_ctx.is_media_changed:
            self.maybe_update_player_position(self.value())
        self._dragging_ctx = None
        self.update_progress(self._app.player.position)

    def on_media_changed(self, media):
        if False:
            for i in range(10):
                print('nop')
        if self._dragging_ctx is not None:
            self._dragging_ctx.is_media_changed = True

    def on_action_triggered(self, action):
        if False:
            print('Hello World!')
        if action not in (QAbstractSlider.SliderNoAction, QAbstractSlider.SliderMove):
            slider_position = self.sliderPosition()
            self.maybe_update_player_position(slider_position)

    def maybe_update_player_position(self, position):
        if False:
            i = 10
            return i + 15
        if self._app.player.current_media:
            self._app.player.position = position
            self._app.player.resume()