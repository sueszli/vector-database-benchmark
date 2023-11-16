"""
 @file
 @brief This file contains the zoom slider QWidget (for interactive zooming/panning on the timeline)
 @author Jonathan Thomas <jonathan@openshot.org>

 @section LICENSE

 Copyright (c) 2008-2018 OpenShot Studios, LLC
 (http://www.openshotstudios.com). This file is part of
 OpenShot Video Editor (http://www.openshot.org), an open-source project
 dedicated to delivering high quality video editing and animation solutions
 to the world.

 OpenShot Video Editor is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 OpenShot Video Editor is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with OpenShot Library.  If not, see <http://www.gnu.org/licenses/>.
 """
from PyQt5.QtCore import Qt, QCoreApplication, QRectF, QTimer
from PyQt5.QtGui import QPainter, QPixmap, QColor, QPen, QBrush, QCursor, QPainterPath, QIcon
from PyQt5.QtWidgets import QSizePolicy, QWidget
import openshot
from classes import updates
from classes.app import get_app
from classes.query import Clip, Track, Transition, Marker

class ZoomSlider(QWidget, updates.UpdateInterface):
    """ A QWidget used to zoom and pan around a Timeline"""

    def changed(self, action):
        if False:
            while True:
                i = 10
        self.clip_rects.clear()
        self.clip_rects_selected.clear()
        self.marker_rects.clear()
        layers = {}
        for (count, layer) in enumerate(reversed(sorted(Track.filter()))):
            layers[layer.data.get('number')] = count
        if hasattr(get_app().window, 'timeline') and self.scrollbar_position[2] != 0.0:
            project_duration = get_app().project.get('duration')
            pixels_per_second = self.width() / project_duration
            vertical_factor = self.height() / len(layers.keys())
            for clip in Clip.filter():
                clip_x = clip.data.get('position', 0.0) * pixels_per_second
                clip_y = layers.get(clip.data.get('layer', 0), 0) * vertical_factor
                clip_width = (clip.data.get('end', 0.0) - clip.data.get('start', 0.0)) * pixels_per_second
                clip_rect = QRectF(clip_x, clip_y, clip_width, 1.0 * vertical_factor)
                if clip.id in get_app().window.selected_clips:
                    self.clip_rects_selected.append(clip_rect)
                else:
                    self.clip_rects.append(clip_rect)
            for clip in Transition.filter():
                clip_x = clip.data.get('position', 0.0) * pixels_per_second
                clip_y = layers.get(clip.data.get('layer', 0), 0) * vertical_factor
                clip_width = (clip.data.get('end', 0.0) - clip.data.get('start', 0.0)) * pixels_per_second
                clip_rect = QRectF(clip_x, clip_y, clip_width, 1.0 * vertical_factor)
                if clip.id in get_app().window.selected_transitions:
                    self.clip_rects_selected.append(clip_rect)
                else:
                    self.clip_rects.append(clip_rect)
            for marker in Marker.filter():
                marker_x = marker.data.get('position', 0.0) * pixels_per_second
                marker_rect = QRectF(marker_x, 0, 0.5, len(layers) * vertical_factor)
                self.marker_rects.append(marker_rect)
        self.update()

    def paintEvent(self, event, *args):
        if False:
            i = 10
            return i + 15
        ' Custom paint event '
        event.accept()
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform | QPainter.TextAntialiasing, True)
        painter.fillRect(event.rect(), QColor('#191919'))
        clip_pen = QPen(QBrush(QColor('#53a0ed')), 1.5)
        clip_pen.setCosmetic(True)
        painter.setPen(clip_pen)
        selected_clip_pen = QPen(QBrush(QColor('Red')), 1.5)
        selected_clip_pen.setCosmetic(True)
        scroll_color = QColor('#4053a0ed')
        scroll_pen = QPen(QBrush(scroll_color), 2.0)
        scroll_pen.setCosmetic(True)
        marker_color = QColor('#4053a0ed')
        marker_pen = QPen(QBrush(marker_color), 1.0)
        marker_pen.setCosmetic(True)
        playhead_color = QColor(Qt.red)
        playhead_color.setAlphaF(0.5)
        playhead_pen = QPen(QBrush(playhead_color), 1.0)
        playhead_pen.setCosmetic(True)
        handle_color = QColor('#a653a0ed')
        handle_pen = QPen(QBrush(handle_color), 1.5)
        handle_pen.setCosmetic(True)
        layers = Track.filter()
        if get_app().window.timeline and self.scrollbar_position[2] != 0.0:
            project_duration = get_app().project.get('duration')
            pixels_per_second = event.rect().width() / project_duration
            project_pixel_width = max(0, project_duration * pixels_per_second)
            scroll_width = (self.scrollbar_position[1] - self.scrollbar_position[0]) * event.rect().width()
            fps_num = get_app().project.get('fps').get('num', 24)
            fps_den = get_app().project.get('fps').get('den', 1) or 1
            fps_float = float(fps_num / fps_den)
            vertical_factor = event.rect().height() / len(layers)
            painter.setPen(clip_pen)
            for clip_rect in self.clip_rects:
                painter.drawRect(clip_rect)
            painter.setPen(selected_clip_pen)
            for clip_rect in self.clip_rects_selected:
                painter.drawRect(clip_rect)
            painter.setPen(marker_pen)
            for marker_rect in self.marker_rects:
                painter.drawRect(marker_rect)
            painter.setPen(playhead_pen)
            playhead_x = self.current_frame / fps_float * pixels_per_second
            playhead_rect = QRectF(playhead_x, 0, 0.5, len(layers) * vertical_factor)
            painter.drawRect(playhead_rect)
            if self.scrollbar_position:
                painter.setPen(scroll_pen)
                scroll_x = self.scrollbar_position[0] * event.rect().width()
                self.scroll_bar_rect = QRectF(scroll_x, 0.0, scroll_width, event.rect().height())
                scroll_path = QPainterPath()
                scroll_path.addRoundedRect(self.scroll_bar_rect, 6, 6)
                painter.fillPath(scroll_path, scroll_color)
                painter.drawPath(scroll_path)
                painter.setPen(handle_pen)
                handle_width = 12.0
                left_handle_x = self.scrollbar_position[0] * event.rect().width() - handle_width / 2.0
                self.left_handle_rect = QRectF(left_handle_x, event.rect().height() / 4.0, handle_width, event.rect().height() / 2.0)
                left_handle_path = QPainterPath()
                left_handle_path.addRoundedRect(self.left_handle_rect, handle_width, handle_width)
                painter.fillPath(left_handle_path, handle_color)
                right_handle_x = self.scrollbar_position[1] * event.rect().width() - handle_width / 2.0
                self.right_handle_rect = QRectF(right_handle_x, event.rect().height() / 4.0, handle_width, event.rect().height() / 2.0)
                right_handle_path = QPainterPath()
                right_handle_path.addRoundedRect(self.right_handle_rect, handle_width, handle_width)
                painter.fillPath(right_handle_path, handle_color)
            if get_app().window.preview_thread.player.Mode() == openshot.PLAYBACK_PLAY and self.is_auto_center:
                if not self.scroll_bar_rect.contains(playhead_rect):
                    get_app().window.TimelineCenter.emit()
        painter.end()

    def mousePressEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        'Capture mouse press event'
        event.accept()
        self.mouse_pressed = True
        self.mouse_dragging = False
        self.mouse_position = event.pos().x()
        self.scrollbar_position_previous = self.scrollbar_position
        get_app().updates.ignore_history = True

    def mouseReleaseEvent(self, event):
        if False:
            i = 10
            return i + 15
        'Capture mouse release event'
        event.accept()
        self.mouse_pressed = False
        self.mouse_dragging = False
        self.left_handle_dragging = False
        self.right_handle_dragging = False
        self.scroll_bar_dragging = False

    def set_handle_limits(self, left_handle, right_handle, is_left=False):
        if False:
            print('Hello World!')
        'Set min/max limits on the bounds of the handles (to prevent invalid values)'
        if left_handle < 0.0:
            left_handle = 0.0
            right_handle = self.scroll_bar_rect.width() / self.width()
        if right_handle > 1.0:
            left_handle = 1.0 - self.scroll_bar_rect.width() / self.width()
            right_handle = 1.0
        diff = right_handle - left_handle
        if is_left and diff < self.min_distance:
            left_handle = right_handle - self.min_distance
        elif not is_left and diff < self.min_distance:
            right_handle = left_handle + self.min_distance
        return (left_handle, right_handle)

    def mouseMoveEvent(self, event):
        if False:
            i = 10
            return i + 15
        'Capture mouse events'
        event.accept()
        mouse_pos = event.pos().x()
        if mouse_pos < 0:
            mouse_pos = 0
        elif mouse_pos > self.width():
            mouse_pos = self.width()
        if not self.mouse_dragging:
            if self.left_handle_rect.contains(event.pos()):
                self.setCursor(self.cursors.get('resize_x'))
            elif self.right_handle_rect.contains(event.pos()):
                self.setCursor(self.cursors.get('resize_x'))
            elif self.scroll_bar_rect.contains(event.pos()):
                self.setCursor(self.cursors.get('move'))
            else:
                self.setCursor(Qt.ArrowCursor)
        if self.mouse_pressed and (not self.mouse_dragging):
            self.mouse_dragging = True
            if self.left_handle_rect.contains(event.pos()):
                self.left_handle_dragging = True
            elif self.right_handle_rect.contains(event.pos()):
                self.right_handle_dragging = True
            elif self.scroll_bar_rect.contains(event.pos()):
                self.scroll_bar_dragging = True
            else:
                self.setCursor(Qt.ArrowCursor)
        if self.mouse_dragging:
            if self.left_handle_dragging:
                delta = (self.mouse_position - mouse_pos) / self.width()
                new_left_pos = self.scrollbar_position_previous[0] - delta
                is_left = True
                if int(QCoreApplication.instance().keyboardModifiers() & Qt.ShiftModifier) > 0:
                    if self.scrollbar_position_previous[1] + delta - new_left_pos > self.min_distance:
                        new_right_pos = self.scrollbar_position_previous[1] + delta
                    else:
                        midpoint = (self.scrollbar_position_previous[1] + self.scrollbar_position_previous) / 2
                        new_right_pos = midpoint + self.min_distance / 2
                        new_left_pos = midpoint - self.min_distance / 2
                else:
                    new_right_pos = self.scrollbar_position_previous[1]
                (new_left_pos, new_right_pos) = self.set_handle_limits(new_left_pos, new_right_pos, is_left)
                self.scrollbar_position = [new_left_pos, new_right_pos, self.scrollbar_position[2], self.scrollbar_position[3]]
                self.delayed_resize_timer.start()
            elif self.right_handle_dragging:
                delta = (self.mouse_position - mouse_pos) / self.width()
                is_left = False
                new_right_pos = self.scrollbar_position_previous[1] - delta
                if int(QCoreApplication.instance().keyboardModifiers() & Qt.ShiftModifier) > 0:
                    if new_right_pos - (self.scrollbar_position_previous[0] + delta) > self.min_distance:
                        new_left_pos = self.scrollbar_position_previous[0] + delta
                    else:
                        midpoint = (self.scrollbar_position_previous[1] + self.scrollbar_position_previous) / 2
                        new_right_pos = midpoint + self.min_distance / 2
                        new_left_pos = midpoint - self.min_distance / 2
                else:
                    new_left_pos = self.scrollbar_position_previous[0]
                (new_left_pos, new_right_pos) = self.set_handle_limits(new_left_pos, new_right_pos, is_left)
                self.scrollbar_position = [new_left_pos, new_right_pos, self.scrollbar_position[2], self.scrollbar_position[3]]
                self.delayed_resize_timer.start()
            elif self.scroll_bar_dragging:
                delta = (self.mouse_position - mouse_pos) / self.width()
                new_left_pos = self.scrollbar_position_previous[0] - delta
                new_right_pos = self.scrollbar_position_previous[1] - delta
                (new_left_pos, new_right_pos) = self.set_handle_limits(new_left_pos, new_right_pos)
                self.scrollbar_position = [new_left_pos, new_right_pos, self.scrollbar_position[2], self.scrollbar_position[3]]
                get_app().window.TimelineScroll.emit(new_left_pos)
            self.update()

    def resizeEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        'Widget resize event'
        event.accept()
        self.delayed_size = self.size()
        self.delayed_resize_timer.start()

    def delayed_resize_callback(self):
        if False:
            for i in range(10):
                print('nop')
        'Callback for resize event timer (to delay the resize event, and prevent lots of similar resize events)'
        project_duration = get_app().project.get('duration')
        normalized_scroll_width = self.scrollbar_position[1] - self.scrollbar_position[0]
        scroll_width_seconds = normalized_scroll_width * project_duration
        tick_pixels = 100
        if self.scrollbar_position[3] > 0.0:
            zoom_factor = scroll_width_seconds / (self.scrollbar_position[3] / tick_pixels)
            if zoom_factor > 0.0:
                self.setZoomFactor(zoom_factor)
                get_app().window.TimelineScroll.emit(self.scrollbar_position[0])

    def wheelEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        event.accept()
        self.repaint()

    def setZoomFactor(self, zoom_factor):
        if False:
            return 10
        'Set the current zoom factor'
        self.zoom_factor = zoom_factor
        get_app().window.TimelineZoom.emit(self.zoom_factor)
        get_app().window.TimelineCenter.emit()
        self.repaint()

    def zoomIn(self):
        if False:
            i = 10
            return i + 15
        'Zoom into timeline'
        if self.zoom_factor >= 10.0:
            new_factor = self.zoom_factor - 5.0
        elif self.zoom_factor >= 4.0:
            new_factor = self.zoom_factor - 2.0
        else:
            new_factor = self.zoom_factor * 0.8
        self.setZoomFactor(new_factor)

    def zoomOut(self):
        if False:
            print('Hello World!')
        'Zoom out of timeline'
        if self.zoom_factor >= 10.0:
            new_factor = self.zoom_factor + 5.0
        elif self.zoom_factor >= 4.0:
            new_factor = self.zoom_factor + 2.0
        else:
            new_factor = min(self.zoom_factor * 1.25, 4.0)
        self.setZoomFactor(new_factor)

    def update_scrollbars(self, new_positions):
        if False:
            print('Hello World!')
        'Consume the current scroll bar positions from the webview timeline'
        if self.mouse_dragging:
            return
        self.scrollbar_position = new_positions
        if not self.clip_rects:
            self.changed(None)
        self.is_auto_center = False
        self.repaint()

    def handle_selection(self):
        if False:
            for i in range(10):
                print('nop')
        self.changed(None)
        self.repaint()

    def update_playhead_pos(self, currentFrame):
        if False:
            print('Hello World!')
        'Callback when position is changed'
        self.current_frame = currentFrame
        self.repaint()

    def handle_play(self):
        if False:
            return 10
        'Callback when play button is clicked'
        self.is_auto_center = True

    def connect_playback(self):
        if False:
            while True:
                i = 10
        'Connect playback signals'
        self.win.preview_thread.position_changed.connect(self.update_playhead_pos)
        self.win.PlaySignal.connect(self.handle_play)

    def __init__(self, *args):
        if False:
            i = 10
            return i + 15
        QWidget.__init__(self, *args)
        _ = get_app()._tr
        self.leftHandle = None
        self.rightHandle = None
        self.centerHandle = None
        self.mouse_pressed = False
        self.mouse_dragging = False
        self.mouse_position = None
        self.zoom_factor = 15.0
        self.scrollbar_position = [0.0, 0.0, 0.0, 0.0]
        self.scrollbar_position_previous = [0.0, 0.0, 0.0, 0.0]
        self.left_handle_rect = QRectF()
        self.left_handle_dragging = False
        self.right_handle_rect = QRectF()
        self.right_handle_dragging = False
        self.scroll_bar_rect = QRectF()
        self.scroll_bar_dragging = False
        self.clip_rects = []
        self.clip_rects_selected = []
        self.marker_rects = []
        self.current_frame = 0
        self.is_auto_center = True
        self.min_distance = 0.02
        self.cursors = {}
        for cursor_name in ['move', 'resize_x', 'hand']:
            icon = QIcon(':/cursors/cursor_%s.png' % cursor_name)
            self.cursors[cursor_name] = QCursor(icon.pixmap(24, 24))
        super().setAttribute(Qt.WA_OpaquePaintEvent)
        super().setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        get_app().updates.add_listener(self)
        self.setMouseTracking(True)
        self.win = get_app().window
        self.win.TimelineScrolled.connect(self.update_scrollbars)
        self.win.TimelineResize.connect(self.delayed_resize_callback)
        self.win.SelectionChanged.connect(self.handle_selection)
        self.delayed_size = None
        self.delayed_resize_timer = QTimer(self)
        self.delayed_resize_timer.setInterval(100)
        self.delayed_resize_timer.setSingleShot(True)
        self.delayed_resize_timer.timeout.connect(self.delayed_resize_callback)