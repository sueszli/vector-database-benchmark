"""
 @file
 @brief This file loads the title editor dialog (i.e SVG creator)
 @author Jonathan Thomas <jonathan@openshot.org>
 @author Andy Finch <andy@openshot.org>

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
import re
import shutil
import sys
import functools
import subprocess
import tempfile
import threading
from xml.dom import minidom
from PyQt5.QtCore import Qt, pyqtSlot, QTimer, pyqtSignal
from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QMessageBox, QDialog, QColorDialog, QFontDialog, QPushButton, QLineEdit, QLabel
import openshot
from classes import info, ui_util
from classes.logger import log
from classes.app import get_app
from classes.metrics import track_metric_screen
from windows.views.titles_listview import TitlesListView
from windows.color_picker import ColorPicker
from classes.style_tools import style_to_dict, dict_to_style, set_if_existing
from windows.views.titles_listview import TitlesListView

class TitleEditor(QDialog):
    """ Title Editor Dialog """
    ui_path = os.path.join(info.PATH, 'windows', 'ui', 'title-editor.ui')
    thumbnailReady = pyqtSignal(object)

    def __init__(self, *args, edit_file_path=None, duplicate=False, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
        self.update_timer = QTimer(self)
        self.update_timer.setInterval(50)
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.save_and_reload)
        self.app = get_app()
        self.project = self.app.project
        self.edit_file_path = edit_file_path
        self.duplicate = duplicate
        ui_util.load_ui(self, self.ui_path)
        ui_util.init_ui(self)
        track_metric_screen('title-screen')
        self.env = dict(os.environ)
        if sys.platform == 'linux':
            self.env.pop('LD_LIBRARY_PATH', None)
            log.debug('Removing custom LD_LIBRARY_PATH from environment variables when launching Inkscape')
        self.is_thread_busy = False
        self.template_name = ''
        imp = minidom.getDOMImplementation()
        self.xmldoc = imp.createDocument(None, 'any', None)
        self.bg_color_code = QtGui.QColor(Qt.black)
        self.font_color_code = QtGui.QColor(Qt.white)
        self.bg_style_string = ''
        self.title_style_string = ''
        self.subtitle_style_string = ''
        self.font_weight = 'normal'
        self.font_style = 'normal'
        self.font_size_ratio = 1
        self.new_title_text = ''
        self.sub_title_text = ''
        self.subTitle = False
        self.display_name = ''
        self.font_family = 'Bitstream Vera Sans'
        self.tspan_nodes = None
        self.qfont = QtGui.QFont(self.font_family)
        self.titlesView = TitlesListView(parent=self, window=self)
        self.verticalLayout.addWidget(self.titlesView)
        self.buttonBox.button(self.buttonBox.Save).setEnabled(False)
        self.thumbnailReady.connect(self.display_pixmap)
        if self.edit_file_path:
            self.widget.setVisible(False)
            self.create_temp_title(self.edit_file_path)
            self.load_svg_template()
            QTimer.singleShot(50, self.display_svg)

    def display_pixmap(self, display_pixmap):
        if False:
            print('Hello World!')
        'Display pixmap of SVG on UI thread'
        self.lblPreviewLabel.setPixmap(display_pixmap)

    def txtLine_changed(self, txtWidget):
        if False:
            return 10
        text_list = []
        for child in self.settingsContainer.children():
            if type(child) == QLineEdit and child.objectName() != 'txtFileName':
                text_list.append(child.text())
        for (i, node) in enumerate(self.tspan_nodes):
            if len(node.childNodes) > 0 and i <= len(text_list) - 1:
                new_text_node = self.xmldoc.createTextNode(text_list[i])
                old_text_node = node.childNodes[0]
                node.removeChild(old_text_node)
                node.appendChild(new_text_node)
        self.update_timer.start()

    def display_svg(self):
        if False:
            i = 10
            return i + 15
        (new_file, tmp_filename) = tempfile.mkstemp(suffix='.png')
        os.close(new_file)
        clip = openshot.Clip(self.filename)
        reader = clip.Reader()
        scale = get_app().devicePixelRatio()
        if scale > 1.0:
            clip.scale_x.AddPoint(1.0, 1.0 * scale)
            clip.scale_y.AddPoint(1.0, 1.0 * scale)
        reader.Open()
        reader.GetFrame(1).Thumbnail(tmp_filename, round(self.lblPreviewLabel.width() * scale), round(self.lblPreviewLabel.height() * scale), '', '', '#000', False, 'png', 85, 0.0)
        reader.Close()
        clip.Close()
        display_pixmap = QtGui.QIcon(tmp_filename).pixmap(self.lblPreviewLabel.size())
        self.thumbnailReady.emit(display_pixmap)
        os.unlink(tmp_filename)

    def create_temp_title(self, template_path):
        if False:
            print('Hello World!')
        'Set temp file path & make copy of template'
        self.filename = os.path.join(info.USER_PATH, 'title', 'temp.svg')
        shutil.copyfile(template_path, self.filename)
        return self.filename

    def load_svg_template(self, filename_field=None):
        if False:
            while True:
                i = 10
        ' Load an SVG title and init all textboxes and controls '
        log.debug('Loading SVG file %s as title template', self.filename)
        _ = get_app()._tr
        layout = self.settingsContainer.layout()
        self.xmldoc = minidom.parse(self.filename)
        self.tspan_nodes = self.xmldoc.getElementsByTagName('tspan')
        self.font_family = 'Bitstream Vera Sans'
        if self.qfont:
            del self.qfont
        self.qfont = QtGui.QFont(self.font_family)
        for child in self.settingsContainer.children():
            try:
                if isinstance(child, QWidget):
                    layout.removeWidget(child)
                    child.deleteLater()
            except Exception as ex:
                log.debug('Failed to delete child settings widget: %s', ex)
        self.text_nodes = self.xmldoc.getElementsByTagName('text')
        self.rect_node = self.xmldoc.getElementsByTagName('rect')
        label = QLabel(self)
        label_line_text = _('File Name:')
        label.setText(label_line_text)
        label.setToolTip(label_line_text)
        self.txtFileName = QLineEdit(self)
        self.txtFileName.setObjectName('txtFileName')
        if filename_field:
            self.txtFileName.setText(filename_field)
        elif self.edit_file_path and (not self.duplicate):
            self.txtFileName.setText(os.path.basename(self.edit_file_path))
            self.txtFileName.setEnabled(False)
        else:
            name = _('TitleFileName (%d)')
            offset = 0
            if self.duplicate and self.edit_file_path:
                name = os.path.basename(self.edit_file_path)
                match = re.match('^(.+?)(\\s*)(\\(([0-9]*)\\))?\\.svg$', name)
                name = match.group(1) + ' (%d)'
                if match.group(4):
                    offset = int(match.group(4))
                    name = match.group(1) + match.group(2) + '(%d)'
            for i in range(1, 1000):
                curname = name % (offset + i)
                possible_path = os.path.join(info.TITLE_PATH, '%s.svg' % curname)
                if not os.path.exists(possible_path):
                    self.txtFileName.setText(curname)
                    break
        self.txtFileName.setFixedHeight(28)
        layout.addRow(label, self.txtFileName)
        title_text = []
        for (i, node) in enumerate(self.tspan_nodes):
            if len(node.childNodes) < 1:
                continue
            text = node.childNodes[0].data
            title_text.append(text)
            s = node.getAttribute('style')
            ard = style_to_dict(s)
            fs = ard.get('font-size')
            if fs and fs.endswith('px'):
                self.qfont.setPixelSize(int(float(fs[:-2])))
            elif fs and fs.endswith('pt'):
                self.qfont.setPointSizeF(float(fs[:-2]))
            label_line_text = _('Line %s:') % str(i + 1)
            label = QLabel(label_line_text)
            label.setToolTip(label_line_text)
            widget = QLineEdit(_(text))
            widget.setFixedHeight(28)
            widget.textChanged.connect(functools.partial(self.txtLine_changed, widget))
            layout.addRow(label, widget)
        label = QLabel(_('Font:'))
        label.setToolTip(_('Font:'))
        self.btnFont = QPushButton(_('Change Font'))
        layout.addRow(label, self.btnFont)
        self.btnFont.clicked.connect(self.btnFont_clicked)
        label = QLabel(_('Text:'))
        label.setToolTip(_('Text:'))
        self.btnFontColor = QPushButton(_('Text Color'))
        layout.addRow(label, self.btnFontColor)
        self.btnFontColor.clicked.connect(self.btnFontColor_clicked)
        label = QLabel(_('Background:'))
        label.setToolTip(_('Background:'))
        self.btnBackgroundColor = QPushButton(_('Background Color'))
        layout.addRow(label, self.btnBackgroundColor)
        self.btnBackgroundColor.clicked.connect(self.btnBackgroundColor_clicked)
        label = QLabel(_('Advanced:'))
        label.setToolTip(_('Advanced:'))
        self.btnAdvanced = QPushButton(_('Use Advanced Editor'))
        layout.addRow(label, self.btnAdvanced)
        self.btnAdvanced.clicked.connect(self.btnAdvanced_clicked)
        self.update_font_color_button()
        self.update_background_color_button()
        if len(title_text) >= 1:
            self.btnFont.setEnabled(True)
            self.btnFontColor.setEnabled(True)
            self.btnBackgroundColor.setEnabled(True)
            self.btnAdvanced.setEnabled(True)
        else:
            self.btnFont.setEnabled(False)
            self.btnFontColor.setEnabled(False)
        self.buttonBox.button(self.buttonBox.Save).setEnabled(True)

    def writeToFile(self, xmldoc):
        if False:
            print('Hello World!')
        'writes a new svg file containing the user edited data'
        if not self.filename.endswith('svg'):
            self.filename = self.filename + '.svg'
        try:
            file = open(os.fsencode(self.filename), 'wb')
            file.write(bytes(xmldoc.toxml(), 'UTF-8'))
            file.close()
        except IOError as inst:
            log.error('Error writing SVG title: {}'.format(inst))

    def save_and_reload(self):
        if False:
            i = 10
            return i + 15
        'Something changed, so update temp SVG and redisplay'
        if not self.is_thread_busy:
            t = threading.Thread(target=self.save_and_reload_thread, daemon=True)
            t.start()
        else:
            self.update_timer.start()

    def save_and_reload_thread(self):
        if False:
            i = 10
            return i + 15
        "Run inside thread, to update and display new SVG - so we don't block the main UI thread"
        self.is_thread_busy = True
        self.writeToFile(self.xmldoc)
        self.display_svg()
        self.is_thread_busy = False

    @pyqtSlot(QtGui.QColor)
    def color_callback(self, save_fn, refresh_fn, color):
        if False:
            i = 10
            return i + 15
        'Update SVG color after user selection'
        if not color or not color.isValid():
            return
        save_fn(color.name(), color.alphaF())
        refresh_fn()
        self.update_timer.start()

    @staticmethod
    def best_contrast(bg: QtGui.QColor) -> QtGui.QColor:
        if False:
            i = 10
            return i + 15
        'Choose text color for best contrast against a background'
        colrgb = bg.getRgbF()
        lum = 0.299 * colrgb[0] + 0.587 * colrgb[1] + 0.114 * colrgb[2]
        if lum < 0.5:
            return QtGui.QColor(Qt.white)
        return QtGui.QColor(Qt.black)

    def btnFontColor_clicked(self):
        if False:
            print('Hello World!')
        app = get_app()
        _ = app._tr
        callback_func = functools.partial(self.color_callback, self.set_font_color_elements, self.update_font_color_button)
        log.debug('Launching color picker for %s', self.font_color_code.name())
        ColorPicker(self.font_color_code, parent=self, title=_('Select a Color'), extra_options=QColorDialog.ShowAlphaChannel, callback=callback_func)

    def btnBackgroundColor_clicked(self):
        if False:
            for i in range(10):
                print('nop')
        app = get_app()
        _ = app._tr
        callback_func = functools.partial(self.color_callback, self.set_bg_style, self.update_background_color_button)
        log.debug('Launching color picker for %s', self.bg_color_code.name())
        ColorPicker(self.bg_color_code, parent=self, title=_('Select a Color'), extra_options=QColorDialog.ShowAlphaChannel, callback=callback_func)

    def btnFont_clicked(self):
        if False:
            i = 10
            return i + 15
        app = get_app()
        _ = app._tr
        oldfont = self.qfont
        (font, ok) = QFontDialog.getFont(oldfont, caption='Change Font')
        if ok and font is not oldfont:
            self.qfont = font
            fontinfo = QtGui.QFontInfo(font)
            oldfontinfo = QtGui.QFontInfo(oldfont)
            self.font_family = fontinfo.family()
            self.font_style = fontinfo.styleName()
            self.font_weight = fontinfo.weight()
            if oldfontinfo.pixelSize() > 0:
                self.font_size_ratio = fontinfo.pixelSize() / oldfontinfo.pixelSize()
            self.set_font_style()
            self.update_timer.start()

    def update_font_color_button(self):
        if False:
            while True:
                i = 10
        'Updates the color shown on the font color button'
        for node in self.text_nodes + self.tspan_nodes:
            s = node.getAttribute('style')
            ard = style_to_dict(s)
            color = ard.get('fill', '#FFF')
            if color.startswith('url(#') and self.xmldoc.getElementsByTagName('defs').length == 1:
                color_ref_id = color[5:-1]
                ref_color = self.get_ref_color(color_ref_id)
                if ref_color:
                    color = ref_color
            opacity = float(ard.get('opacity', 1.0))
            color = QtGui.QColor(color)
            text_color = self.best_contrast(color)
            self.btnFontColor.setStyleSheet('background-color: %s; opacity: %s; color: %s;' % (color.name(), 1, text_color.name()))
            color.setAlphaF(opacity)
            self.font_color_code = color
            log.debug('Set color of font-color button to %s', color.name())

    def get_ref_color(self, id):
        if False:
            return 10
        'Get the color value from a reference id (i.e. linearGradient3267)'
        for ref_node in self.xmldoc.getElementsByTagName('defs')[0].childNodes:
            if ref_node.attributes and 'id' in ref_node.attributes:
                ref_node_id = ref_node.attributes['id'].value
                if id == ref_node_id:
                    if 'xlink:href' in ref_node.attributes:
                        xlink_ref_id = ref_node.attributes['xlink:href'].value[1:]
                        return self.get_ref_color(xlink_ref_id)
                    if 'href' in ref_node.attributes:
                        xlink_ref_id = ref_node.attributes['href'].value[1:]
                        return self.get_ref_color(xlink_ref_id)
                    elif ref_node.childNodes:
                        for stop_node in ref_node.childNodes:
                            if stop_node.nodeName == 'stop':
                                ard = style_to_dict(stop_node.getAttribute('style'))
                                if 'stop-color' in ard:
                                    return ard.get('stop-color')
        return ''

    def update_background_color_button(self):
        if False:
            while True:
                i = 10
        'Updates the color shown on the background color button'
        if self.rect_node:
            s = self.rect_node[0].getAttribute('style')
            ard = style_to_dict(s)
            color = ard.get('fill', '#000')
            opacity = float(ard.get('opacity', 1.0))
            color = QtGui.QColor(color)
            text_color = self.best_contrast(color)
            self.btnBackgroundColor.setStyleSheet('background-color: %s; opacity: %s; color: %s;' % (color.name(), 1, text_color.name()))
            color.setAlphaF(opacity)
            self.bg_color_code = color
            log.debug('Set color of background-color button to %s', color.name())

    def set_font_style(self):
        if False:
            for i in range(10):
                print('nop')
        'sets the font properties'
        for text_child in self.text_nodes + self.tspan_nodes:
            s = text_child.getAttribute('style')
            ard = style_to_dict(s)
            set_if_existing(ard, 'font-style', self.font_style)
            set_if_existing(ard, 'font-family', f"'{self.font_family}'")
            new_font_size_pixel = 0
            if 'font-size' in ard:
                new_font_size_pixel = self.font_size_ratio * float(ard['font-size'][:-2])
            set_if_existing(ard, 'font-size', f'{new_font_size_pixel}px')
            self.title_style_string = dict_to_style(ard)
            text_child.setAttribute('style', self.title_style_string)
        log.debug('Updated font styles to %s', self.title_style_string)

    def set_bg_style(self, color, alpha):
        if False:
            i = 10
            return i + 15
        'sets the background color'
        if self.rect_node:
            s = self.rect_node[0].getAttribute('style')
            ard = style_to_dict(s)
            ard.update({'fill': color, 'opacity': str(alpha)})
            self.bg_style_string = dict_to_style(ard)
            self.rect_node[0].setAttribute('style', self.bg_style_string)
            log.debug('Updated background style to %s', self.bg_style_string)

    def set_font_color_elements(self, color, alpha):
        if False:
            print('Hello World!')
        for text_child in self.text_nodes + self.tspan_nodes:
            s = text_child.getAttribute('style')
            ard = style_to_dict(s)
            ard.update({'fill': color, 'opacity': str(alpha)})
            text_child.setAttribute('style', dict_to_style(ard))
        log.debug('Set text node style, fill:%s opacity:%s', color, alpha)

    def accept(self):
        if False:
            return 10
        app = get_app()
        _ = app._tr
        if self.edit_file_path and (not self.duplicate):
            self.filename = self.edit_file_path
            self.writeToFile(self.xmldoc)
        else:
            file_name = '%s.svg' % self.txtFileName.text().strip()
            file_path = os.path.join(info.TITLE_PATH, file_name)
            if self.txtFileName.text().strip():
                if os.path.exists(file_path) and (not self.edit_file_path):
                    ret = QMessageBox.question(self, _('Title Editor'), _('%s already exists.\nDo you want to replace it?') % file_name, QMessageBox.No | QMessageBox.Yes)
                    if ret == QMessageBox.No:
                        return
                self.filename = file_path
                self.writeToFile(self.xmldoc)
                app.window.files_model.add_files(self.filename, prevent_image_seq=True, prevent_recent_folder=True)
        super().accept()

    def btnAdvanced_clicked(self):
        if False:
            print('Hello World!')
        'Use an external editor to edit the image'
        s = get_app().get_settings()
        prog = s.get('title_editor')
        filename_text = self.txtFileName.text().strip()
        try:
            log.info('Advanced title editor command: %s', str([prog, self.filename]))
            p = subprocess.Popen([prog, self.filename], env=self.env)
            p.communicate()
            self.load_svg_template(filename_field=filename_text)
            self.display_svg()
        except OSError:
            _ = self.app._tr
            msg = QMessageBox(self)
            msg.setText(_('Please install %s to use this function' % prog))
            msg.exec_()