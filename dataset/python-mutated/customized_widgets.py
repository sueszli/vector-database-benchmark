"""
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
try:
    from PySide6.QtWidgets import QDateTimeEdit
    from PySide6.QtCore import QSettings, Qt
except:
    from PyQt5.QtWidgets import QDateTimeEdit
    from PyQt5.QtCore import QSettings, Qt
persepolis_setting = QSettings('persepolis_download_manager', 'persepolis')
ui_direction = persepolis_setting.value('ui_direction')

class MyQDateTimeEdit(QDateTimeEdit):

    def __init__(self, parent=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent)
        if ui_direction == 'rtl':
            self.setLayoutDirection(Qt.LeftToRight)