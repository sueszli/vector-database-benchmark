"""
 @file
 @brief This file contains the properties tableview, used by the main window
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
import os
import json
import functools
from operator import itemgetter
from PyQt5.QtCore import Qt, QRectF, QLocale, pyqtSignal, pyqtSlot, QRect
from PyQt5.QtGui import QCursor, QIcon, QColor, QBrush, QPen, QPalette, QPixmap, QPainter, QPainterPath, QLinearGradient, QFont, QFontInfo
from PyQt5.QtWidgets import QTableView, QAbstractItemView, QMenu, QSizePolicy, QHeaderView, QItemDelegate, QStyle, QLabel, QPushButton, QHBoxLayout, QFrame, QFontDialog
from classes.logger import log
from classes.app import get_app
from classes import info
from classes.query import Clip, Effect, Transition
from windows.models.properties_model import PropertiesModel
from windows.color_picker import ColorPicker
import openshot

class PropertyDelegate(QItemDelegate):

    def __init__(self, parent=None, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.model = kwargs.pop('model', None)
        if not self.model:
            log.error('Cannot create delegate without data model!')
        super().__init__(parent, *args, **kwargs)
        self.curve_pixmaps = {openshot.BEZIER: QIcon(':/curves/keyframe-%s.png' % openshot.BEZIER).pixmap(20, 20), openshot.LINEAR: QIcon(':/curves/keyframe-%s.png' % openshot.LINEAR).pixmap(20, 20), openshot.CONSTANT: QIcon(':/curves/keyframe-%s.png' % openshot.CONSTANT).pixmap(20, 20)}

    def paint(self, painter, option, index):
        if False:
            i = 10
            return i + 15
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        model = self.model
        row = model.itemFromIndex(index).row()
        selected_label = model.item(row, 0)
        selected_value = model.item(row, 1)
        cur_property = selected_label.data()
        property_type = cur_property[1]['type']
        property_max = cur_property[1]['max']
        property_min = cur_property[1]['min']
        readonly = cur_property[1]['readonly']
        points = cur_property[1]['points']
        interpolation = cur_property[1]['interpolation']
        if property_type in ['float', 'int']:
            current_value = QLocale().system().toDouble(selected_value.text())[0]
            if property_min < 0.0:
                property_shift = 0.0 - property_min
                property_min += property_shift
                property_max += property_shift
                current_value += property_shift
            min_max_range = float(property_max) - float(property_min)
            value_percent = current_value / min_max_range
        else:
            value_percent = 0.0
        painter.setPen(QPen(Qt.NoPen))
        if property_type == 'color':
            red = int(cur_property[1]['red']['value'])
            green = int(cur_property[1]['green']['value'])
            blue = int(cur_property[1]['blue']['value'])
            painter.setBrush(QColor(red, green, blue))
        elif option.state & QStyle.State_Selected:
            painter.setBrush(QColor('#575757'))
        else:
            painter.setBrush(QColor('#3e3e3e'))
        if readonly:
            painter.setPen(QPen(get_app().window.palette().color(QPalette.Disabled, QPalette.Text)))
        else:
            path = QPainterPath()
            path.addRoundedRect(QRectF(option.rect), 15, 15)
            painter.fillPath(path, QColor('#3e3e3e'))
            painter.drawPath(path)
            painter.setBrush(QBrush(QColor('#000000')))
            mask_rect = QRectF(option.rect)
            mask_rect.setWidth(option.rect.width() * value_percent)
            painter.setClipRect(mask_rect, Qt.IntersectClip)
            gradient = QLinearGradient(option.rect.topLeft(), option.rect.topRight())
            gradient.setColorAt(0, QColor('#828282'))
            gradient.setColorAt(1, QColor('#828282'))
            painter.setBrush(gradient)
            path = QPainterPath()
            value_rect = QRectF(option.rect)
            path.addRoundedRect(value_rect, 15, 15)
            painter.fillPath(path, gradient)
            painter.drawPath(path)
            painter.setClipping(False)
            if points > 1:
                painter.drawPixmap(int(option.rect.x() + option.rect.width() - 30.0), int(option.rect.y() + 4), self.curve_pixmaps[interpolation])
            painter.setPen(QPen(Qt.white))
        value = index.data(Qt.DisplayRole)
        if value:
            painter.drawText(option.rect, Qt.AlignCenter, value)
        painter.restore()

class PropertiesTableView(QTableView):
    """ A Properties Table QWidget used on the main window """
    loadProperties = pyqtSignal(str, str)

    def mouseMoveEvent(self, event):
        if False:
            i = 10
            return i + 15
        model = self.clip_properties_model.model
        if self.lock_selection and self.prev_row:
            row = self.prev_row
        else:
            row = self.indexAt(event.pos()).row()
            self.prev_row = row
            self.lock_selection = True
        if row is None:
            return
        event.accept()
        if model.item(row, 0):
            self.selected_label = model.item(row, 0)
            self.selected_item = model.item(row, 1)
        if self.selected_label and self.selected_item and self.selected_label.data() and (type(self.selected_label.data()) == tuple):
            get_app().updates.ignore_history = True
            openshot.Settings.Instance().ENABLE_PLAYBACK_CACHING = False
            log.debug('mouseMoveEvent: Stop caching frames on timeline')
            value_column_x = self.columnViewportPosition(1)
            cursor_value = event.x() - value_column_x
            cursor_value_percent = cursor_value / self.columnWidth(1)
            try:
                cur_property = self.selected_label.data()
            except Exception:
                log.debug('Failed to access data on selected label widget')
                return
            if type(cur_property) != tuple:
                log.debug('Failed to access valid data on current selected label widget')
                return
            property_key = cur_property[0]
            property_name = cur_property[1]['name']
            property_type = cur_property[1]['type']
            property_max = cur_property[1]['max']
            property_min = cur_property[1]['min']
            readonly = cur_property[1]['readonly']
            (item_id, item_type) = self.selected_item.data()
            if readonly:
                return
            if not self.original_data:
                c = None
                if item_type == 'clip':
                    c = Clip.get(id=item_id)
                elif item_type == 'transition':
                    c = Transition.get(id=item_id)
                elif item_type == 'effect':
                    c = Effect.get(id=item_id)
                if c and property_key in c.data:
                    self.original_data = c.data
            if property_type in ['float', 'int'] and property_name != 'Track':
                if self.previous_x == -1:
                    self.diff_length = 10
                    self.previous_x = event.x()
                drag_diff = self.previous_x - event.x()
                self.previous_x = event.x()
                if abs(drag_diff) < self.diff_length:
                    self.diff_length = max(0, self.diff_length - 1)
                    return
                min_max_range = float(property_max) - float(property_min)
                if min_max_range < 1000.0:
                    self.new_value = property_min + min_max_range * cursor_value_percent
                else:
                    self.new_value = QLocale().system().toDouble(self.selected_item.text())[0]
                    if drag_diff > 0:
                        self.new_value -= 0.5
                    elif drag_diff < 0:
                        self.new_value += 0.5
                self.new_value = max(property_min, self.new_value)
                self.new_value = min(property_max, self.new_value)
                if property_type == 'int':
                    self.new_value = round(self.new_value, 0)
                self.clip_properties_model.value_updated(self.selected_item, -1, self.new_value)
                self.viewport().update()

    def mouseReleaseEvent(self, event):
        if False:
            return 10
        event.accept()
        get_app().updates.ignore_history = False
        openshot.Settings.Instance().ENABLE_PLAYBACK_CACHING = True
        log.debug('mouseReleaseEvent: Start caching frames on timeline')
        get_app().updates.apply_last_action_to_history(self.original_data)
        self.original_data = None
        model = self.clip_properties_model.model
        row = self.indexAt(event.pos()).row()
        if model.item(row, 0):
            self.selected_label = model.item(row, 0)
            self.selected_item = model.item(row, 1)
        self.lock_selection = False
        self.previous_x = -1

    @pyqtSlot(QColor)
    def color_callback(self, newColor: QColor):
        if False:
            for i in range(10):
                print('nop')
        if newColor.isValid():
            self.clip_properties_model.color_update(self.selected_item, newColor)

    def doubleClickedCB(self, model_index):
        if False:
            i = 10
            return i + 15
        'Double click handler for the property table'
        _ = get_app()._tr
        model = self.clip_properties_model.model
        row = model_index.row()
        selected_label = model.item(row, 0)
        self.selected_item = model.item(row, 1)
        if selected_label and selected_label.data() and (type(selected_label.data()) == tuple):
            cur_property = selected_label.data()
            property_type = cur_property[1]['type']
            if property_type == 'color':
                red = cur_property[1]['red']['value']
                green = cur_property[1]['green']['value']
                blue = cur_property[1]['blue']['value']
                currentColor = QColor(int(red), int(green), int(blue))
                log.debug('Launching ColorPicker for %s', currentColor.name())
                ColorPicker(currentColor, parent=self, title=_('Select a Color'), callback=self.color_callback)
                return
            elif property_type == 'font':
                current_font_name = cur_property[1].get('memo', 'sans')
                current_font = QFont(current_font_name)
                (font, ok) = QFontDialog.getFont(current_font, caption='Change Font')
                if ok and font:
                    fontinfo = QFontInfo(font)
                    font_details = {'font_family': fontinfo.family(), 'font_style': fontinfo.styleName(), 'font_weight': fontinfo.weight(), 'font_size_pixel': fontinfo.pixelSize()}
                    self.clip_properties_model.value_updated(self.selected_item, value=fontinfo.family())

    def caption_text_updated(self, new_caption_text, caption_model_row):
        if False:
            while True:
                i = 10
        'Caption text has been updated in the caption editor, and needs saving'
        if not caption_model_row:
            return
        cur_property = caption_model_row[0].data()
        property_type = cur_property[1]['type']
        if property_type == 'caption' and cur_property[1].get('memo') != new_caption_text:
            self.clip_properties_model.value_updated(caption_model_row[1], value=new_caption_text)

    def select_item(self, item_id, item_type):
        if False:
            print('Hello World!')
        ' Update the selected item in the properties window '
        _ = get_app()._tr
        self.clip_properties_model.update_item(item_id, item_type)

    def select_frame(self, frame_number):
        if False:
            while True:
                i = 10
        ' Update the values of the selected clip, based on the current frame '
        self.clip_properties_model.update_frame(frame_number)

    def filter_changed(self, value=None):
        if False:
            print('Hello World!')
        ' Filter the list of properties '
        self.clip_properties_model.update_model(value)
        get_app().window.SetKeyframeFilter.emit(value)

    def contextMenuEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        ' Display context menu '
        index = self.indexAt(event.pos())
        if not index.isValid():
            event.ignore()
            return
        idx = self.indexAt(event.pos())
        row = idx.row()
        selected_label = idx.model().item(row, 0)
        selected_value = idx.model().item(row, 1)
        self.selected_item = selected_value
        frame_number = self.clip_properties_model.frame_number
        _ = get_app()._tr
        if selected_label and selected_label.data() and (type(selected_label.data()) == tuple):
            cur_property = selected_label.data()
            if self.menu_reset:
                self.choices = []
                self.menu_reset = False
            property_name = cur_property[1]['name']
            self.property_type = cur_property[1]['type']
            points = cur_property[1]['points']
            self.choices = cur_property[1]['choices']
            property_key = cur_property[0]
            (clip_id, item_type) = selected_value.data()
            log.info('Context menu shown for %s (%s) for clip %s on frame %s' % (property_name, property_key, clip_id, frame_number))
            log.info('Points: %s' % points)
            if property_key == 'parent_effect_id' and (not self.choices):
                effect = Effect.get(id=clip_id)
                clip_choices = []
                for clip in Clip.filter():
                    file_id = clip.data.get('file_id')
                    parent_clip_id = effect.parent.get('id')
                    if clip.id != parent_clip_id:
                        for file_index in range(self.files_model.rowCount()):
                            file_row = self.files_model.index(file_index, 0)
                            project_file_id = file_row.sibling(file_index, 5).data()
                            if file_id == project_file_id:
                                clip_instance_icon = file_row.data(Qt.DecorationRole)
                                break
                        effect_choices = []
                        for clip_effect_data in clip.data['effects']:
                            if clip_effect_data['class_name'] == effect.data['class_name']:
                                effect_id = clip_effect_data['id']
                                effect_icon = QIcon(QPixmap(os.path.join(info.PATH, 'effects', 'icons', '%s.png' % clip_effect_data['class_name'].lower())))
                                effect_choices.append({'name': effect_id, 'value': effect_id, 'selected': False, 'icon': effect_icon})
                        if effect_choices:
                            clip_choices.append({'name': _(clip.data['title']), 'value': effect_choices, 'selected': False, 'icon': clip_instance_icon})
                self.choices.append({'name': _('None'), 'value': 'None', 'selected': False, 'icon': None})
                if clip_choices:
                    self.choices.append({'name': _('Clips'), 'value': clip_choices, 'selected': False, 'icon': None})
            if property_key == 'selected_object_index' and (not self.choices):
                timeline_instance = get_app().window.timeline_sync.timeline
                effect = timeline_instance.GetClipEffect(clip_id)
                visible_objects = json.loads(effect.GetVisibleObjects(frame_number))
                object_index_choices = []
                for object_index in visible_objects['visible_objects_index']:
                    object_index_choices.append({'name': str(object_index), 'value': str(object_index), 'selected': False, 'icon': None})
                if object_index_choices:
                    self.choices.append({'name': _('Detected Objects'), 'value': object_index_choices, 'selected': False, 'icon': None})
            if property_key in ['parentObjectId', 'child_clip_id'] and (not self.choices):
                tracked_choices = []
                clip_choices = []
                timeline_instance = get_app().window.timeline_sync.timeline
                for clip in Clip.filter():
                    file_id = clip.data.get('file_id')
                    parent_clip_id = clip_id
                    if item_type == 'effect':
                        parent_clip_id = Effect.get(id=clip_id).parent.get('id')
                        log.debug(f"Lookup parent clip ID for effect: '{clip_id}' = '{parent_clip_id}'")
                    if clip.id != parent_clip_id:
                        for file_index in range(self.files_model.rowCount()):
                            file_row = self.files_model.index(file_index, 0)
                            project_file_id = file_row.sibling(file_index, 5).data()
                            if file_id == project_file_id:
                                clip_instance_icon = file_row.data(Qt.DecorationRole)
                                clip_choices.append({'name': clip.data['title'], 'value': clip.id, 'selected': False, 'icon': clip_instance_icon})
                        icon_size = 72
                        icon_pixmap = clip_instance_icon.pixmap(icon_size, icon_size)
                        tracked_objects = []
                        for effect in clip.data['effects']:
                            if effect.get('has_tracked_object'):
                                effect_instance = timeline_instance.GetClipEffect(effect['id'])
                                visible_objects_id = json.loads(effect_instance.GetVisibleObjects(frame_number))['visible_objects_id']
                                for object_id in visible_objects_id:
                                    object_properties = json.loads(timeline_instance.GetTrackedObjectValues(object_id, 0))
                                    x1 = object_properties['x1']
                                    y1 = object_properties['y1']
                                    x2 = object_properties['x2']
                                    y2 = object_properties['y2']
                                    tracked_object_icon = icon_pixmap.copy(QRect(x1 * icon_size, y1 * icon_size, (x2 - x1) * icon_size, (y2 - y1) * icon_size)).scaled(icon_size, icon_size)
                                    tracked_objects.append({'name': str(object_id), 'value': str(object_id), 'selected': False, 'icon': QIcon(tracked_object_icon)})
                            tracked_choices.append({'name': clip.data['title'], 'value': tracked_objects, 'selected': False, 'icon': clip_instance_icon})
                self.choices.append({'name': _('None'), 'value': 'None', 'selected': False, 'icon': None})
                if property_key == 'parentObjectId' and tracked_choices:
                    self.choices.append({'name': _('Tracked Objects'), 'value': tracked_choices, 'selected': False, 'icon': None})
                if clip_choices:
                    self.choices.append({'name': _('Clips'), 'value': clip_choices, 'selected': False, 'icon': None})
            if self.property_type == 'reader' and (not self.choices):
                file_choices = []
                for i in range(self.files_model.rowCount()):
                    idx = self.files_model.index(i, 0)
                    if not idx.isValid():
                        continue
                    icon = idx.data(Qt.DecorationRole)
                    name = idx.sibling(i, 1).data()
                    path = os.path.join(idx.sibling(i, 4).data(), name)
                    file_choices.append({'name': name, 'value': path, 'selected': False, 'icon': icon})
                if file_choices:
                    self.choices.append({'name': _('Files'), 'value': file_choices, 'selected': False, icon: None})
                trans_choices = []
                for i in range(self.transition_model.rowCount()):
                    idx = self.transition_model.index(i, 0)
                    if not idx.isValid():
                        continue
                    icon = idx.data(Qt.DecorationRole)
                    name = idx.sibling(i, 1).data()
                    path = idx.sibling(i, 3).data()
                    trans_choices.append({'name': name, 'value': path, 'selected': False, 'icon': icon})
                self.choices.append({'name': _('Transitions'), 'value': trans_choices, 'selected': False})
            if property_name == 'Track' and self.property_type == 'int' and (not self.choices):
                all_tracks = get_app().project.get('layers')
                display_count = len(all_tracks)
                for track in reversed(sorted(all_tracks, key=itemgetter('number'))):
                    track_name = track.get('label') or _('Track %s') % display_count
                    self.choices.append({'name': track_name, 'value': track.get('number'), 'selected': False, 'icon': None})
                    display_count -= 1
                return
            elif self.property_type == 'font':
                current_font_name = cur_property[1].get('memo', 'sans')
                current_font = QFont(current_font_name)
                (font, ok) = QFontDialog.getFont(current_font, caption='Change Font')
                if ok and font:
                    fontinfo = QFontInfo(font)
                    self.clip_properties_model.value_updated(self.selected_item, value=fontinfo.family())
            bezier_presets = [(0.25, 0.1, 0.25, 1.0, _('Ease (Default)')), (0.42, 0.0, 1.0, 1.0, _('Ease In')), (0.0, 0.0, 0.58, 1.0, _('Ease Out')), (0.42, 0.0, 0.58, 1.0, _('Ease In/Out')), (0.55, 0.085, 0.68, 0.53, _('Ease In (Quad)')), (0.55, 0.055, 0.675, 0.19, _('Ease In (Cubic)')), (0.895, 0.03, 0.685, 0.22, _('Ease In (Quart)')), (0.755, 0.05, 0.855, 0.06, _('Ease In (Quint)')), (0.47, 0.0, 0.745, 0.715, _('Ease In (Sine)')), (0.95, 0.05, 0.795, 0.035, _('Ease In (Expo)')), (0.6, 0.04, 0.98, 0.335, _('Ease In (Circ)')), (0.6, -0.28, 0.735, 0.045, _('Ease In (Back)')), (0.25, 0.46, 0.45, 0.94, _('Ease Out (Quad)')), (0.215, 0.61, 0.355, 1.0, _('Ease Out (Cubic)')), (0.165, 0.84, 0.44, 1.0, _('Ease Out (Quart)')), (0.23, 1.0, 0.32, 1.0, _('Ease Out (Quint)')), (0.39, 0.575, 0.565, 1.0, _('Ease Out (Sine)')), (0.19, 1.0, 0.22, 1.0, _('Ease Out (Expo)')), (0.075, 0.82, 0.165, 1.0, _('Ease Out (Circ)')), (0.175, 0.885, 0.32, 1.275, _('Ease Out (Back)')), (0.455, 0.03, 0.515, 0.955, _('Ease In/Out (Quad)')), (0.645, 0.045, 0.355, 1.0, _('Ease In/Out (Cubic)')), (0.77, 0.0, 0.175, 1.0, _('Ease In/Out (Quart)')), (0.86, 0.0, 0.07, 1.0, _('Ease In/Out (Quint)')), (0.445, 0.05, 0.55, 0.95, _('Ease In/Out (Sine)')), (1.0, 0.0, 0.0, 1.0, _('Ease In/Out (Expo)')), (0.785, 0.135, 0.15, 0.86, _('Ease In/Out (Circ)')), (0.68, -0.55, 0.265, 1.55, _('Ease In/Out (Back)'))]
            menu = QMenu(self)
            if self.property_type == 'color':
                Color_Action = menu.addAction(_('Select a Color'))
                Color_Action.triggered.connect(functools.partial(self.Color_Picker_Triggered, cur_property))
                menu.addSeparator()
            if points > 1:
                Bezier_Menu = menu.addMenu(self.bezier_icon, _('Bezier'))
                for bezier_preset in bezier_presets:
                    preset_action = Bezier_Menu.addAction(bezier_preset[4])
                    preset_action.triggered.connect(functools.partial(self.Bezier_Action_Triggered, bezier_preset))
                Linear_Action = menu.addAction(self.linear_icon, _('Linear'))
                Linear_Action.triggered.connect(self.Linear_Action_Triggered)
                Constant_Action = menu.addAction(self.constant_icon, _('Constant'))
                Constant_Action.triggered.connect(self.Constant_Action_Triggered)
                menu.addSeparator()
            if points >= 1:
                Insert_Action = menu.addAction(_('Insert Keyframe'))
                Insert_Action.triggered.connect(self.Insert_Action_Triggered)
                Remove_Action = menu.addAction(_('Remove Keyframe'))
                Remove_Action.triggered.connect(self.Remove_Action_Triggered)
                menu.popup(event.globalPos())
            log.debug(f'Context menu choices: {self.choices}')
            if not self.choices:
                return
            for choice in self.choices:
                if type(choice['value']) != list:
                    Choice_Action = menu.addAction(_(choice['name']))
                    Choice_Action.setData(choice['value'])
                    Choice_Action.triggered.connect(self.Choice_Action_Triggered)
                    continue
                SubMenu = None
                if choice.get('icon') is not None:
                    SubMenuRoot = menu.addMenu(choice['icon'], choice['name'])
                else:
                    SubMenuRoot = menu.addMenu(choice['name'])
                SubMenuSize = 25
                SubMenuNumber = 0
                if len(choice['value']) > SubMenuSize:
                    SubMenu = SubMenuRoot.addMenu(str(SubMenuNumber))
                else:
                    SubMenu = SubMenuRoot
                for (i, sub_choice) in enumerate(choice['value'], 1):
                    if type(sub_choice['value']) == list:
                        SubSubMenu = SubMenu.addMenu(sub_choice['icon'], sub_choice['name'])
                        for sub_sub_choice in sub_choice['value']:
                            Choice_Action = SubSubMenu.addAction(sub_sub_choice['icon'], sub_sub_choice['name'])
                            Choice_Action.setData(sub_sub_choice['value'])
                            Choice_Action.triggered.connect(self.Choice_Action_Triggered)
                    else:
                        if i % SubMenuSize == 0:
                            SubMenuNumber += 1
                            SubMenu = SubMenuRoot.addMenu(str(SubMenuNumber))
                        Choice_Action = SubMenu.addAction(sub_choice['icon'], _(sub_choice['name']))
                        Choice_Action.setData(sub_choice['value'])
                        Choice_Action.triggered.connect(self.Choice_Action_Triggered)
            log.debug(f'Display context menu: {menu.children()}')
            menu.popup(event.globalPos())

    def Bezier_Action_Triggered(self, preset=[]):
        if False:
            print('Hello World!')
        log.info('Bezier_Action_Triggered: %s' % str(preset))
        if self.property_type != 'color':
            self.clip_properties_model.value_updated(self.selected_item, interpolation=0, interpolation_details=preset)
        else:
            self.clip_properties_model.color_update(self.selected_item, QColor('#000'), interpolation=0, interpolation_details=preset)

    def Linear_Action_Triggered(self):
        if False:
            print('Hello World!')
        log.info('Linear_Action_Triggered')
        if self.property_type != 'color':
            self.clip_properties_model.value_updated(self.selected_item, interpolation=1)
        else:
            self.clip_properties_model.color_update(self.selected_item, QColor('#000'), interpolation=1, interpolation_details=[])

    def Constant_Action_Triggered(self):
        if False:
            return 10
        log.info('Constant_Action_Triggered')
        if self.property_type != 'color':
            self.clip_properties_model.value_updated(self.selected_item, interpolation=2)
        else:
            self.clip_properties_model.color_update(self.selected_item, QColor('#000'), interpolation=2, interpolation_details=[])

    def Color_Picker_Triggered(self, cur_property):
        if False:
            print('Hello World!')
        log.info('Color_Picker_Triggered')
        _ = get_app()._tr
        red = int(cur_property[1]['red']['value'])
        green = int(cur_property[1]['green']['value'])
        blue = int(cur_property[1]['blue']['value'])
        currentColor = QColor(red, green, blue)
        log.debug('Launching ColorPicker for %s', currentColor.name())
        ColorPicker(currentColor, parent=self, title=_('Select a Color'), callback=self.color_callback)

    def Insert_Action_Triggered(self):
        if False:
            for i in range(10):
                print('nop')
        log.info('Insert_Action_Triggered')
        if self.selected_item:
            current_value = QLocale().system().toDouble(self.selected_item.text())[0]
            self.clip_properties_model.value_updated(self.selected_item, value=current_value)

    def Remove_Action_Triggered(self):
        if False:
            i = 10
            return i + 15
        log.info('Remove_Action_Triggered')
        self.clip_properties_model.remove_keyframe(self.selected_item)

    def Choice_Action_Triggered(self):
        if False:
            for i in range(10):
                print('nop')
        log.info('Choice_Action_Triggered')
        choice_value = self.sender().data()
        self.clip_properties_model.value_updated(self.selected_item, value=choice_value)

    def refresh_menu(self):
        if False:
            print('Hello World!')
        ' Ensure we update the menu when our source models change '
        self.menu_reset = True

    def __init__(self, *args):
        if False:
            i = 10
            return i + 15
        QTableView.__init__(self, *args)
        self.win = get_app().window
        self.clip_properties_model = PropertiesModel(self)
        self.transition_model = self.win.transition_model.model
        self.files_model = self.win.files_model.model
        self.files_model.dataChanged.connect(self.refresh_menu)
        self.win.transition_model.ModelRefreshed.connect(self.refresh_menu)
        self.menu_reset = False
        self.selected = []
        self.selected_label = None
        self.selected_item = None
        self.new_value = None
        self.original_data = None
        self.lock_selection = False
        self.prev_row = None
        self.bezier_icon = QIcon(QPixmap(os.path.join(info.IMAGES_PATH, 'keyframe-%s.png' % openshot.BEZIER)))
        self.linear_icon = QIcon(QPixmap(os.path.join(info.IMAGES_PATH, 'keyframe-%s.png' % openshot.LINEAR)))
        self.constant_icon = QIcon(QPixmap(os.path.join(info.IMAGES_PATH, 'keyframe-%s.png' % openshot.CONSTANT)))
        self.setModel(self.clip_properties_model.model)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setWordWrap(True)
        delegate = PropertyDelegate(model=self.clip_properties_model.model)
        self.setItemDelegateForColumn(1, delegate)
        self.previous_x = -1
        horizontal_header = self.horizontalHeader()
        horizontal_header.setSectionResizeMode(QHeaderView.Stretch)
        vertical_header = self.verticalHeader()
        vertical_header.setVisible(False)
        self.clip_properties_model.update_model()
        self.resizeColumnToContents(0)
        self.resizeColumnToContents(1)
        get_app().window.txtPropertyFilter.textChanged.connect(self.filter_changed)
        get_app().window.InsertKeyframe.connect(self.Insert_Action_Triggered)
        self.doubleClicked.connect(self.doubleClickedCB)
        self.loadProperties.connect(self.select_item)
        get_app().window.CaptionTextUpdated.connect(self.caption_text_updated)

class SelectionLabel(QFrame):
    """ The label to display selections """

    def getMenu(self):
        if False:
            return 10
        menu = QMenu(self)
        _ = get_app()._tr
        if self.item_type == 'clip':
            item = Clip.get(id=self.item_id)
            if item:
                self.item_name = item.title()
        elif self.item_type == 'transition':
            item = Transition.get(id=self.item_id)
            if item:
                self.item_name = item.title()
        elif self.item_type == 'effect':
            item = Effect.get(id=self.item_id)
            if item:
                self.item_name = item.title()
        if not self.item_name:
            return
        for item_id in get_app().window.selected_clips:
            clip = Clip.get(id=item_id)
            if clip:
                item_name = clip.title()
                item_icon = QIcon(QPixmap(clip.data.get('image')))
                action = menu.addAction(item_icon, item_name)
                action.setData({'item_id': item_id, 'item_type': 'clip'})
                action.triggered.connect(self.Action_Triggered)
                for effect in clip.data.get('effects'):
                    effect = Effect.get(id=effect.get('id'))
                    if effect:
                        item_name = effect.title()
                        item_icon = QIcon(QPixmap(os.path.join(info.PATH, 'effects', 'icons', '%s.png' % effect.data.get('class_name').lower())))
                        action = menu.addAction(item_icon, '  >  %s' % _(item_name))
                        action.setData({'item_id': effect.id, 'item_type': 'effect'})
                        action.triggered.connect(self.Action_Triggered)
        for item_id in get_app().window.selected_transitions:
            trans = Transition.get(id=item_id)
            if trans:
                item_name = _(trans.title())
                item_icon = QIcon(QPixmap(trans.data.get('reader', {}).get('path')))
                action = menu.addAction(item_icon, _(item_name))
                action.setData({'item_id': item_id, 'item_type': 'transition'})
                action.triggered.connect(self.Action_Triggered)
        for item_id in get_app().window.selected_effects:
            effect = Effect.get(id=item_id)
            if effect:
                item_name = _(effect.title())
                item_icon = QIcon(QPixmap(os.path.join(info.PATH, 'effects', 'icons', '%s.png' % effect.data.get('class_name').lower())))
                action = menu.addAction(item_icon, _(item_name))
                action.setData({'item_id': item_id, 'item_type': 'effect'})
                action.triggered.connect(self.Action_Triggered)
        return menu

    def Action_Triggered(self):
        if False:
            i = 10
            return i + 15
        item_id = self.sender().data()['item_id']
        item_type = self.sender().data()['item_type']
        log.info('switch selection to %s:%s' % (item_id, item_type))
        get_app().window.propertyTableView.loadProperties.emit(item_id, item_type)

    def select_item(self, item_id, item_type):
        if False:
            return 10
        self.item_name = None
        self.item_icon = None
        self.item_type = item_type
        self.item_id = item_id
        _ = get_app()._tr
        if self.item_type == 'clip':
            clip = Clip.get(id=self.item_id)
            if clip:
                self.item_name = clip.title()
                self.item_icon = QIcon(QPixmap(clip.data.get('image')))
        elif self.item_type == 'transition':
            trans = Transition.get(id=self.item_id)
            if trans:
                self.item_name = _(trans.title())
                self.item_icon = QIcon(QPixmap(trans.data.get('reader', {}).get('path')))
        elif self.item_type == 'effect':
            effect = Effect.get(id=self.item_id)
            if effect:
                self.item_name = _(effect.title())
                self.item_icon = QIcon(QPixmap(os.path.join(info.PATH, 'effects', 'icons', '%s.png' % effect.data.get('class_name').lower())))
        if self.item_name and len(self.item_name) > 25:
            self.item_name = '%s...' % self.item_name[:22]
        if self.item_id:
            self.lblSelection.setText('<strong>%s</strong>' % _('Selection:'))
            self.btnSelectionName.setText(self.item_name)
            self.btnSelectionName.setVisible(True)
            if self.item_icon:
                self.btnSelectionName.setIcon(self.item_icon)
        else:
            self.lblSelection.setText('<strong>%s</strong>' % _('No Selection'))
            self.btnSelectionName.setVisible(False)
        self.btnSelectionName.setMenu(self.getMenu())

    def __init__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args)
        self.item_id = None
        self.item_type = None
        _ = get_app()._tr
        self.lblSelection = QLabel()
        self.lblSelection.setText('<strong>%s</strong>' % _('No Selection'))
        self.btnSelectionName = QPushButton()
        self.btnSelectionName.setVisible(False)
        self.btnSelectionName.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.lblSelection.setTextFormat(Qt.RichText)
        hbox = QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.addWidget(self.lblSelection)
        hbox.addWidget(self.btnSelectionName)
        self.setLayout(hbox)
        get_app().window.propertyTableView.loadProperties.connect(self.select_item)