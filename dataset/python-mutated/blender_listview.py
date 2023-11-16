"""
 @file
 @brief This file contains the blender file listview, used by the 3d animated titles screen
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
import copy
import subprocess
import sys
import re
import functools
import shlex
import json
from time import sleep
try:
    from defusedxml import minidom as xml
except ImportError:
    from xml.dom import minidom as xml
from PyQt5.QtCore import Qt, QObject, pyqtSlot, pyqtSignal, QThread, QTimer, QSize
from PyQt5.QtWidgets import QApplication, QListView, QMessageBox, QComboBox, QDoubleSpinBox, QLabel, QPushButton, QLineEdit, QPlainTextEdit
from PyQt5.QtGui import QColor, QImage, QPixmap, QIcon
from classes import info
from classes.logger import log
from classes.query import File
from classes.app import get_app
from windows.models.blender_model import BlenderModel
from windows.color_picker import ColorPicker

class BlenderListView(QListView):
    """ A ListView QWidget used on the animated title window """
    start_render = pyqtSignal(str, str, int)

    def currentChanged(self, selected, deselected):
        if False:
            for i in range(10):
                print('nop')
        self.selected = selected
        self.deselected = deselected
        _ = self.app._tr
        self.win.clear_effect_controls()
        animation = self.get_animation_details()
        self.selected_template = animation.get('service')
        if not self.selected_template:
            return
        self.generateUniqueFolder()
        for param in animation.get('params', []):
            log.debug('Using parameter %s: %s' % (param['name'], param['title']))
            if param['name'] in ['start_frame', 'end_frame']:
                self.params[param['name']] = int(param['default'])
                continue
            widget = None
            label = QLabel()
            label.setText(_(param['title']))
            label.setToolTip(_(param['title']))
            if param['type'] == 'spinner':
                self.params[param['name']] = float(param['default'])
                widget = QDoubleSpinBox()
                widget.setMinimum(float(param['min']))
                widget.setMaximum(float(param['max']))
                widget.setValue(float(param['default']))
                widget.setSingleStep(0.01)
                widget.setToolTip(param['title'])
                widget.valueChanged.connect(functools.partial(self.spinner_value_changed, param))
            elif param['type'] == 'text':
                self.params[param['name']] = _(param['default'])
                widget = QLineEdit()
                widget.setText(_(param['default']))
                widget.textChanged.connect(functools.partial(self.text_value_changed, widget, param))
            elif param['type'] == 'multiline':
                self.params[param['name']] = _(param['default'])
                widget = QPlainTextEdit()
                widget.setPlainText(_(param['default']).replace('\\n', '\n'))
                widget.textChanged.connect(functools.partial(self.text_value_changed, widget, param))
            elif param['type'] == 'dropdown':
                self.params[param['name']] = param['default']
                widget = QComboBox()
                widget.currentIndexChanged.connect(functools.partial(self.dropdown_index_changed, widget, param))
                if 'project_files' in param['name']:
                    param['values'] = {}
                    for file in File.filter():
                        if file.data['media_type'] not in ('image', 'video'):
                            continue
                        fileName = os.path.basename(file.data['path'])
                        fileExtension = os.path.splitext(fileName)[1]
                        if fileExtension.lower() in '.svg':
                            continue
                        param['values'][fileName] = '|'.join((file.data['path'], str(file.data['height']), str(file.data['width']), file.data['media_type'], str(file.data['fps']['num'] / file.data['fps']['den'])))
                for (i, (k, v)) in enumerate(sorted(param['values'].items())):
                    widget.addItem(_(k), v)
                    if v == param['default']:
                        widget.setCurrentIndex(i)
                if not param['values']:
                    widget.addItem(_('No Files Found'), '')
                    widget.setEnabled(False)
            elif param['type'] == 'color':
                color = QColor(param['default'])
                self.params[param['name']] = [color.redF(), color.greenF(), color.blueF()]
                if 'diffuse_color' in param.get('name'):
                    self.params[param['name']].append(color.alphaF())
                widget = QPushButton()
                widget.setText('')
                widget.setStyleSheet('background-color: {}'.format(param['default']))
                widget.clicked.connect(functools.partial(self.color_button_clicked, widget, param))
            if widget and label:
                self.win.settingsContainer.layout().addRow(label, widget)
            elif label:
                self.win.settingsContainer.layout().addRow(label)
        self.end_processing()
        self.init_slider_values()

    def spinner_value_changed(self, param, value):
        if False:
            while True:
                i = 10
        self.params[param['name']] = value
        log.info('Animation param %s set to %s' % (param['name'], value))

    def text_value_changed(self, widget, param, value=None):
        if False:
            for i in range(10):
                print('nop')
        try:
            if not value:
                value = widget.toPlainText()
        except Exception:
            log.debug('Failed to read plain text value from widget')
            return
        self.params[param['name']] = value

    def dropdown_index_changed(self, widget, param, index):
        if False:
            while True:
                i = 10
        value = widget.itemData(index)
        self.params[param['name']] = value
        log.info('Animation param %s set to %s' % (param['name'], value))
        if param['name'] == 'length_multiplier':
            self.params[param['name']] = float(value) * self.project_fps_diff
            self.init_slider_values()

    def color_button_clicked(self, widget, param, index):
        if False:
            for i in range(10):
                print('nop')
        _ = get_app()._tr
        color_value = self.params[param['name']]
        currentColor = QColor('#FFFFFF')
        if len(color_value) >= 3:
            currentColor.setRgbF(color_value[0], color_value[1], color_value[2])
        self._color_scratchpad = (widget, param)
        ColorPicker(currentColor, callback=self.color_selected, parent=self.win)

    @pyqtSlot(QColor)
    def color_selected(self, newColor):
        if False:
            i = 10
            return i + 15
        'Callback when the user chooses a color in the dialog'
        if not self._color_scratchpad:
            log.warning('ColorPicker callback called without parameter to set')
            return
        (widget, param) = self._color_scratchpad
        if not newColor or not newColor.isValid():
            return
        widget.setStyleSheet('background-color: {}'.format(newColor.name()))
        self.params[param['name']] = [newColor.redF(), newColor.greenF(), newColor.blueF()]
        if 'diffuse_color' in param.get('name'):
            self.params[param['name']].append(newColor.alphaF())
        log.info('Animation param %s set to %s', param['name'], newColor.name())

    def generateUniqueFolder(self):
        if False:
            return 10
        ' Generate a new, unique folder name to contain Blender frames '
        self.unique_folder_name = str(self.app.project.generate_id())
        if not os.path.exists(os.path.join(info.BLENDER_PATH, self.unique_folder_name)):
            os.mkdir(os.path.join(info.BLENDER_PATH, self.unique_folder_name))

    def processing_mode(self, cursor=True):
        if False:
            return 10
        ' Disable all controls on interface '
        self.focus_owner = self.win.focusWidget()
        self.win.btnRefresh.setEnabled(False)
        self.win.sliderPreview.setEnabled(False)
        self.win.btnRender.setEnabled(False)
        if cursor:
            QApplication.setOverrideCursor(Qt.WaitCursor)

    @pyqtSlot()
    def end_processing(self):
        if False:
            while True:
                i = 10
        ' Enable all controls on interface '
        self.win.btnRefresh.setEnabled(True)
        self.win.sliderPreview.setEnabled(True)
        self.win.btnRender.setEnabled(True)
        self.win.statusContainer.hide()
        QApplication.restoreOverrideCursor()
        if self.focus_owner:
            self.focus_owner.setFocus()

    def init_slider_values(self):
        if False:
            while True:
                i = 10
        ' Init the slider and preview frame label to the currently selected animation '
        length = int(self.params.get('end_frame', 1) * self.params.get('length_multiplier', 1.0))
        middle_frame = int(length / 2)
        self.win.sliderPreview.setMinimum(self.params.get('start_frame', 1))
        self.win.sliderPreview.setMaximum(length)
        self.win.sliderPreview.setValue(middle_frame)
        self.preview_timer.start()

    @pyqtSlot()
    def render_finished(self):
        if False:
            return 10
        if not self.final_render:
            return
        filename = '{}%04d.png'.format(self.params['file_name'])
        seq_params = {'folder_path': os.path.join(info.BLENDER_PATH, self.unique_folder_name), 'base_name': self.params['file_name'], 'fixlen': True, 'digits': 4, 'extension': 'png', 'fps': {'num': self.fps.get('num', 25), 'den': self.fps.get('den', 1)}, 'pattern': filename, 'path': os.path.join(os.path.join(info.BLENDER_PATH, self.unique_folder_name), filename)}
        log.info('RENDER FINISHED! Adding to project files: {}'.format(filename))
        get_app().window.files_model.add_files(seq_params.get('path'), seq_params, prevent_recent_folder=True)
        self.win.close()

    @pyqtSlot(str)
    def render_stage(self, stage=None):
        if False:
            print('Hello World!')
        _ = get_app()._tr
        self.win.frameProgress.setRange(0, 0)
        self.win.frameStatus.setText(_('Generating'))
        log.debug('Set Blender progress to Generating step')

    @pyqtSlot(int, int)
    def render_progress(self, step_value, step_max):
        if False:
            print('Hello World!')
        _ = get_app()._tr
        self.win.frameProgress.setRange(0, step_max)
        self.win.frameProgress.setValue(step_value)
        self.win.frameStatus.setText(_('Rendering'))
        log.debug('set Blender progress to Rendering step, %d of %d complete', step_value, step_max)

    @pyqtSlot(int)
    def render_saved(self, frame=None):
        if False:
            for i in range(10):
                print('nop')
        _ = get_app()._tr
        self.win.frameProgress.setValue(self.win.frameProgress.maximum() + 1)
        self.win.frameStatus.setText(_('Saved'))
        log.debug('Set Blender progress to Saved step')

    @pyqtSlot()
    def render_initialize(self):
        if False:
            i = 10
            return i + 15
        _ = get_app()._tr
        self.win.frameProgress.setRange(0, 0)
        self.win.frameStatus.setText(_('Initializing'))
        self.win.statusContainer.show()
        log.debug('Set Blender progress to Initializing step')

    @pyqtSlot(int)
    def update_progress_bar(self, current_frame):
        if False:
            for i in range(10):
                print('nop')
        self.win.sliderPreview.setValue(current_frame)
        length = int(self.params.get('end_frame', 1) * self.params.get('length_multiplier', 1.0))
        self.win.lblFrame.setText('{}/{}'.format(current_frame, length))

    @pyqtSlot(int)
    def sliderPreview_valueChanged(self, new_value):
        if False:
            i = 10
            return i + 15
        'Get new value of preview slider, and start timer to Render frame'
        if self.win.sliderPreview.isEnabled():
            self.preview_timer.start()
        length = int(self.params.get('end_frame', 1) * self.params.get('length_multiplier', 1.0))
        self.win.lblFrame.setText('{}/{}'.format(new_value, length))

    def preview_timer_onTimeout(self):
        if False:
            i = 10
            return i + 15
        'Timer is ready to Render frame'
        preview_frame_number = self.win.sliderPreview.value()
        log.info('Previewing frame %s' % preview_frame_number)
        self.Render(preview_frame_number)

    def get_animation_details(self):
        if False:
            while True:
                i = 10
        ' Build a dictionary of all animation settings and properties from XML '
        current = self.selectionModel().currentIndex()
        if not current.isValid():
            return {}
        animation_title = current.sibling(current.row(), 1).data(Qt.DisplayRole)
        xml_path = current.sibling(current.row(), 2).data(Qt.DisplayRole)
        service = current.sibling(current.row(), 3).data(Qt.DisplayRole)
        xmldoc = xml.parse(xml_path)
        animation = {'title': animation_title, 'path': xml_path, 'service': service, 'params': []}
        for param in xmldoc.getElementsByTagName('param'):
            param_item = {'default': ''}
            for att in ['title', 'description', 'name', 'type']:
                if param.attributes[att]:
                    param_item[att] = param.attributes[att].value
            for tag in ['min', 'max', 'step', 'digits', 'default']:
                for p in param.getElementsByTagName(tag):
                    if p.childNodes:
                        param_item[tag] = p.firstChild.data
            try:
                param_item['values'] = dict([(p.attributes['name'].value, p.attributes['num'].value) for p in param.getElementsByTagName('value') if 'name' in p.attributes and 'num' in p.attributes])
            except (TypeError, AttributeError) as ex:
                log.warn('XML parser: %s', ex)
                pass
            animation['params'].append(param_item)
        xmldoc.unlink()
        return animation

    def mousePressEvent(self, event):
        if False:
            while True:
                i = 10
        event.ignore()
        super().mousePressEvent(event)

    def refresh_view(self):
        if False:
            for i in range(10):
                print('nop')
        self.blender_model.update_model()
        self.blender_model.proxy_model.sort(0)

    def get_project_params(self, is_preview=True):
        if False:
            i = 10
            return i + 15
        ' Return a dictionary of project related settings, needed by the Blender python script. '
        project = self.app.project
        project_params = {}
        fps = project.get('fps')
        project_params['fps'] = fps['num']
        if fps['den'] != 1:
            project_params['fps_base'] = fps['den']
        project_params['resolution_x'] = project.get('width')
        project_params['resolution_y'] = project.get('height')
        if is_preview:
            project_params['resolution_percentage'] = 50
        else:
            project_params['resolution_percentage'] = 100
        project_params['quality'] = 100
        project_params['file_format'] = 'PNG'
        project_params['color_mode'] = 'RGBA'
        project_params['alpha_mode'] = 1
        project_params['horizon_color'] = (0.57, 0.57, 0.57)
        project_params['animation'] = True
        project_params['output_path'] = os.path.join(info.BLENDER_PATH, self.unique_folder_name, self.params['file_name'])
        return project_params

    @pyqtSlot(str)
    def onBlenderVersionError(self, version):
        if False:
            print('Hello World!')
        self.error_with_blender(version, None)

    @pyqtSlot()
    @pyqtSlot(str)
    def onBlenderError(self, error=None):
        if False:
            for i in range(10):
                print('nop')
        self.error_with_blender(None, error)

    def error_with_blender(self, version=None, worker_message=None):
        if False:
            while True:
                i = 10
        ' Show a friendly error message regarding the blender executable or version. '
        _ = self.app._tr
        s = self.app.get_settings()
        error_message = ''
        if version:
            error_message = _('Version Detected: {}').format(version)
            log.info('Blender version detected: {}'.format(version))
        if worker_message:
            error_message = _('Error Output:\n{}').format(worker_message)
            log.error('Blender error: {}'.format(worker_message))
        QMessageBox.critical(self, error_message, _("\nBlender, the free open source 3D content creation suite, is required for this action. (http://www.blender.org)\n\nPlease check the preferences in OpenShot and be sure the Blender executable is correct.\nThis setting should be the path of the 'blender' executable on your computer.\nAlso, please be sure that it is pointing to Blender version {} or greater.\n\nBlender Path: {}\n{}").format(info.BLENDER_MIN_VERSION, s.get('blender_command'), error_message))
        self.win.close()

    def inject_params(self, source_path, out_path, frame=None):
        if False:
            for i in range(10):
                print('nop')
        is_preview = False
        if frame:
            is_preview = True
        user_params = '\n#BEGIN INJECTING PARAMS\n'
        param_data = json.loads(json.dumps(self.params))
        param_data.update(self.get_project_params(is_preview))
        param_serialization = json.dumps(param_data)
        user_params += 'params_json = r' + '"""{}"""'.format(param_serialization)
        user_params += '\n#END INJECTING PARAMS\n'
        s = self.app.get_settings()
        gpu_code_body = None
        if s.get('blender_gpu_enabled'):
            gpu_enable_py = os.path.join(info.PATH, 'blender', 'scripts', 'gpu_enable.py.in')
            try:
                with open(gpu_enable_py, 'r') as f:
                    gpu_code_body = f.read()
                if gpu_code_body:
                    log.info('Injecting GPU enable code from {}'.format(gpu_enable_py))
                    user_params += '\n#ENABLE GPU RENDERING\n'
                    user_params += gpu_code_body
                    user_params += '\n#END ENABLE GPU RENDERING\n'
            except IOError as e:
                log.error('Could not load GPU enable code! %s', e)
        with open(source_path, 'r') as f:
            script_body = f.read()
        script_body = script_body.replace('# INJECT_PARAMS_HERE', user_params)
        try:
            with open(out_path, 'w', encoding='UTF-8', errors='strict') as f:
                f.write(script_body)
        except Exception:
            log.error('Could not write blender script to %s', out_path, exc_info=1)

    @pyqtSlot(str)
    def update_image(self, image_path):
        if False:
            while True:
                i = 10
        scale = get_app().devicePixelRatio()
        display_pixmap = QIcon(image_path).pixmap(self.win.imgPreview.size())
        display_pixmap.setDevicePixelRatio(scale)
        self.win.imgPreview.setPixmap(display_pixmap)

    def Cancel(self):
        if False:
            print('Hello World!')
        'Cancel the current render, if any'
        if 'worker' in dir(self):
            self.worker.Cancel()

    def Render(self, frame=None):
        if False:
            i = 10
            return i + 15
        ' Render an images sequence of the current template using Blender 2.62+ and the\n        Blender Python API. '
        self.processing_mode()
        blend_file_path = os.path.join(info.PATH, 'blender', 'blend', self.selected_template)
        source_script = os.path.join(info.PATH, 'blender', 'scripts', self.selected_template.replace('.blend', '.py.in'))
        target_script = os.path.join(info.BLENDER_PATH, self.unique_folder_name, self.selected_template.replace('.blend', '.py'))
        self.background = QThread(self)
        self.background.setObjectName('openshot_renderer')
        self.worker = Worker(blend_file_path, target_script, int(frame or 0))
        self.worker.setObjectName('render_worker')
        self.worker.moveToThread(self.background)
        self.background.started.connect(self.worker.Render)
        self.worker.render_complete.connect(self.render_finished)
        self.worker.end_processing.connect(self.end_processing)
        self.worker.start_processing.connect(self.render_initialize)
        self.worker.blender_version_error.connect(self.onBlenderVersionError)
        self.worker.blender_error_nodata.connect(self.onBlenderError)
        self.worker.blender_error_with_data.connect(self.onBlenderError)
        self.worker.progress.connect(self.update_progress_bar)
        self.worker.image_updated.connect(self.update_image)
        self.worker.frame_saved.connect(self.render_saved)
        self.worker.frame_stage.connect(self.render_stage)
        self.worker.frame_render.connect(self.render_progress)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.finished.connect(self.background.quit, Qt.DirectConnection)
        self.background.finished.connect(self.background.deleteLater)
        self.background.finished.connect(self.worker.deleteLater)
        self.inject_params(source_script, target_script, frame)
        self.final_render = frame is None
        self.background.start()

    def __init__(self, parent, *args):
        if False:
            while True:
                i = 10
        super().__init__(*args)
        self.win = parent
        self.app = get_app()
        self.blender_model = BlenderModel()
        self.selected = None
        self.deselected = None
        self._color_scratchpad = None
        self.selected_template = ''
        self.final_render = False
        self.preview_timer = QTimer(self)
        self.preview_timer.setInterval(300)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self.preview_timer_onTimeout)
        self.fps = self.app.project.get('fps')
        fps_float = self.fps['num'] / float(self.fps['den'])
        self.project_fps_diff = round(fps_float / 25.0)
        self.params = {}
        self.unique_folder_name = None
        self.processing_mode(cursor=False)
        self.setModel(self.blender_model.proxy_model)
        self.setIconSize(info.LIST_ICON_SIZE)
        self.setGridSize(info.LIST_GRID_SIZE)
        self.setViewMode(QListView.IconMode)
        self.setResizeMode(QListView.Adjust)
        self.setUniformItemSizes(True)
        self.setWordWrap(True)
        self.setTextElideMode(Qt.ElideRight)
        self.win.btnRefresh.clicked.connect(self.preview_timer.start)
        self.win.sliderPreview.valueChanged.connect(functools.partial(self.sliderPreview_valueChanged))
        self.refresh_view()

class Worker(QObject):
    """ Background Worker Object (to run the Blender commands) """
    finished = pyqtSignal()
    blender_version_error = pyqtSignal(str)
    blender_error_nodata = pyqtSignal()
    blender_error_with_data = pyqtSignal(str)
    progress = pyqtSignal(int)
    image_updated = pyqtSignal(str)
    frame_stage = pyqtSignal(str)
    frame_render = pyqtSignal(int, int)
    frame_saved = pyqtSignal(int)
    start_processing = pyqtSignal()
    end_processing = pyqtSignal()
    render_complete = pyqtSignal()

    def __init__(self, blend_file_path, target_script, preview_frame=0):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.blend_file_path = blend_file_path
        self.target_script = target_script
        self.preview_frame = preview_frame
        s = get_app().get_settings()
        self.blender_exec_path = s.get('blender_command')
        self.blender_version_re = re.compile('Blender ([0-9a-z\\.]*)', flags=re.MULTILINE)
        self.blender_frame_re = re.compile('Fra:([0-9,]+)')
        self.blender_saved_re = re.compile("Saved: '(.*\\.png)")
        self.blender_syncing_re = re.compile('\\| Syncing (.*)$', flags=re.MULTILINE)
        self.blender_rendering_re = re.compile('Rendering ([0-9]*) / ([0-9]*) samples')
        self.version = None
        self.process = None
        self.canceled = False
        self.env = dict(os.environ)
        if sys.platform == 'linux':
            self.env.pop('LD_LIBRARY_PATH', None)
            log.debug('Removing custom LD_LIBRARY_PATH from environment variables when launching Blender')
        self.startupinfo = None
        if sys.platform == 'win32':
            self.startupinfo = subprocess.STARTUPINFO()
            self.startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

    def Cancel(self):
        if False:
            i = 10
            return i + 15
        'Cancel worker render'
        if self.process:
            while self.process and self.process.poll() == None:
                log.debug('Terminating Blender Process')
                self.process.terminate()
                sleep(0.1)
        self.canceled = True

    def blender_version_check(self):
        if False:
            for i in range(10):
                print('nop')
        command_get_version = [self.blender_exec_path, '--factory-startup', '-v']
        log.debug('Checking Blender version, command: {}'.format(' '.join([shlex.quote(x) for x in command_get_version])))
        try:
            if self.process:
                self.process.terminate()
            self.process = subprocess.Popen(command_get_version, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, startupinfo=self.startupinfo, env=self.env)
            (out, err) = self.process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.blender_error_nodata.emit()
            return False
        except Exception:
            log.error('Version check exception', exc_info=1)
            self.blender_error_nodata.emit()
            return False
        ver_string = out.decode('utf-8')
        log.debug('Blender output:\n%s', ver_string)
        ver_match = self.blender_version_re.search(ver_string)
        if not ver_match:
            raise Exception('No Blender version detected in output')
        log.debug('Matched %s in output', str(ver_match.group(0)))
        self.version = ver_match.group(1)
        log.info('Found Blender version {}'.format(self.version))
        if self.version < info.BLENDER_MIN_VERSION:
            self.blender_version_error.emit(self.version)
        return self.version >= info.BLENDER_MIN_VERSION

    def process_line(self, out_line):
        if False:
            return 10
        line = out_line.decode('utf-8').strip()
        if not line:
            return
        self.command_output += line + '\n'
        log.debug('  {}'.format(line))
        output_frame = self.blender_frame_re.search(line)
        if output_frame and self.current_frame != int(output_frame.group(1)):
            self.current_frame = int(output_frame.group(1))
            self.progress.emit(self.current_frame)
        output_syncing = self.blender_syncing_re.search(line)
        if output_syncing:
            self.frame_stage.emit(output_syncing.group(1))
        output_rendering = self.blender_rendering_re.search(line)
        if output_rendering:
            self.frame_render.emit(int(output_rendering.group(1)), int(output_rendering.group(2)))
        output_saved = self.blender_saved_re.search(line)
        if output_saved:
            self.frame_count += 1
            log.debug('Saved frame %d', self.current_frame)
            self.frame_saved.emit(self.current_frame)
            self.image_updated.emit(output_saved.group(1))

    @pyqtSlot()
    def Render(self):
        if False:
            for i in range(10):
                print('nop')
        " Worker's Render method which invokes the Blender rendering commands "
        _ = get_app()._tr
        if not self.version and (not self.blender_version_check()):
            self.finished.emit()
            return
        self.command_output = ''
        self.current_frame = 0
        self.frame_count = 0
        try:
            command_render = [self.blender_exec_path, '--factory-startup', '-b', self.blend_file_path, '-y', '-P', self.target_script]
            if self.preview_frame > 0:
                command_render.extend(['-f', str(self.preview_frame)])
            else:
                command_render.extend(['-a'])
            log.debug('Running Blender, command: {}'.format(' '.join([shlex.quote(x) for x in command_render])))
            log.debug('Blender output:')
            if self.process:
                self.process.terminate()
            self.process = subprocess.Popen(command_render, bufsize=512, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, startupinfo=self.startupinfo, env=self.env)
            self.start_processing.emit()
        except subprocess.SubprocessError as ex:
            self.blender_error_with_data.emit(str(ex))
            raise
        except Exception:
            log.error('Worker exception', exc_info=1)
            return
        else:
            while not self.canceled and self.process.poll() is None:
                for out_line in iter(self.process.stdout.readline, b''):
                    self.process_line(out_line)
            self.end_processing.emit()
            log.info('Blender process exited, %d frames saved.', self.frame_count)
            if self.frame_count < 1:
                log.warning('No frame detected from Blender!')
                log.warning('Blender output:\n{}'.format(self.command_output))
                self.blender_error_with_data.emit(_('No frame was found in the output from Blender'))
            else:
                self.render_complete.emit()
        finally:
            self.finished.emit()