"""
 @file
 @brief WebEngine backend for TimelineWebView
 @author Jonathan Thomas <jonathan@openshot.org>
 @author FeRD (Frank Dana) <ferdnyc@gmail.com>

 @section LICENSE

 Copyright (c) 2008-2020 OpenShot Studios, LLC
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
import logging
from functools import partial
from classes import info
from classes.logger import log
from PyQt5.QtCore import QFileInfo, QUrl, Qt, QTimer
from PyQt5.QtGui import QColor
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEnginePage
from PyQt5.QtWebChannel import QWebChannel

class LoggingWebEnginePage(QWebEnginePage):
    """Override console.log message to display messages"""

    def javaScriptConsoleMessage(self, level, msg, line, source):
        if False:
            print('Hello World!')
        log.log(self.levels[level], '%s@L%d: %s', os.path.basename(source), line, msg)

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        super().__init__(parent=parent)
        self.setObjectName('LoggingWebEnginePage')
        self.levels = [logging.INFO, logging.WARNING, logging.ERROR]

class TimelineWebEngineView(QWebEngineView):
    """QtWebEngine Timeline Widget"""

    def __init__(self):
        if False:
            i = 10
            return i + 15
        'Initialization code required for widget'
        super().__init__()
        self.setObjectName('TimelineWebEngineView')
        self.document_is_ready = False
        self.html_path = os.path.join(info.PATH, 'timeline', 'index.html')
        self.new_page = LoggingWebEnginePage(self)
        self.setPage(self.new_page)
        self.page().setBackgroundColor(QColor('#363636'))
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.settings().setAttribute(self.settings().ScrollAnimatorEnabled, True)
        self.webchannel = QWebChannel(self.page())
        self.setHtml(self.get_html(), QUrl.fromLocalFile(QFileInfo(self.html_path).absoluteFilePath()))
        self.page().setWebChannel(self.webchannel)
        log.info('WebEngine backend initializing')
        self.page().loadStarted.connect(self.setup_js_data)

    def run_js(self, code, callback=None, retries=0):
        if False:
            while True:
                i = 10
        'Run JS code async and optionally have a callback for response'
        if not self.document_is_ready:
            if retries == 0:
                log.debug('run_js() called before document ready event. Script queued: %s', code)
            elif retries % 5 == 0:
                log.warning('WebEngine backend still not ready after %d retries.', retries)
            else:
                log.debug('Script queued, %d retries so far', retries)
            QTimer.singleShot(200, partial(self.run_js, code, callback, retries + 1))
            return None
        if callback:
            return self.page().runJavaScript(code, callback)
        return self.page().runJavaScript(code)

    def setup_js_data(self):
        if False:
            for i in range(10):
                print('nop')
        log.info('Registering WebChannel connection with WebEngine')
        self.webchannel.registerObject('timeline', self)

    def get_html(self):
        if False:
            i = 10
            return i + 15
        'Get HTML for Timeline, adjusted for mixin'
        with open(self.html_path, 'r', encoding='utf-8') as f:
            html = f.read()
        return html.replace('<!--MIXIN_JS_INCLUDE-->', '\n                <script type="text/javascript" src="js/mixin_webengine.js"></script>\n            ')

    def keyPressEvent(self, event):
        if False:
            i = 10
            return i + 15
        ' Keypress callback for timeline '
        key_value = event.key()
        if key_value in [Qt.Key_Shift, Qt.Key_Control]:
            return QWebEngineView.keyPressEvent(self, event)
        event.ignore()