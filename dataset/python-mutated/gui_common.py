"""
OnionShare | https://onionshare.org/

Copyright (C) 2014-2022 Micah Lee, et al. <micah@micahflee.com>

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
import os
import shutil
from pkg_resources import resource_filename
from PySide6 import QtCore, QtWidgets, QtGui
from . import strings
from onionshare_cli.onion import Onion, TorErrorInvalidSetting, TorErrorAutomatic, TorErrorSocketPort, TorErrorSocketFile, TorErrorMissingPassword, TorErrorUnreadableCookieFile, TorErrorAuthError, TorErrorProtocolError, BundledTorTimeout, BundledTorBroken, TorTooOldEphemeral, TorTooOldStealth, PortNotAvailable
from onionshare_cli.meek import Meek
from onionshare_cli.web.web import WaitressException

class GuiCommon:
    """
    The shared code for all of the OnionShare GUI.
    """
    MODE_SHARE = 'share'
    MODE_RECEIVE = 'receive'
    MODE_WEBSITE = 'website'
    MODE_CHAT = 'chat'

    def __init__(self, common, qtapp, local_only):
        if False:
            while True:
                i = 10
        self.common = common
        self.qtapp = qtapp
        self.local_only = local_only
        self.is_flatpak = os.path.exists('/.flatpak-info')
        self.common.load_settings()
        strings.load_strings(self.common, self.get_resource_path('locale'))
        self.onion = Onion(common, get_tor_paths=self.get_tor_paths)
        self.lock_filename = os.path.join(self.common.build_data_dir(), 'lock')
        self.events_dir = os.path.join(self.common.build_data_dir(), 'events')
        if not os.path.exists(self.events_dir):
            os.makedirs(self.events_dir, 448, True)
        self.events_filename = os.path.join(self.events_dir, 'events')
        self.meek = Meek(self.common, get_tor_paths=self.get_tor_paths)
        self.css = self.get_css(qtapp.color_mode)
        self.color_mode = qtapp.color_mode

    def get_css(self, color_mode):
        if False:
            print('Hello World!')
        header_color = '#4E064F'
        title_color = '#333333'
        stop_button_color = '#d0011b'
        new_tab_button_background = '#ffffff'
        new_tab_button_border = '#efeff0'
        new_tab_button_text_color = '#4e0d4e'
        downloads_uploads_progress_bar_border_color = '#4E064F'
        downloads_uploads_progress_bar_chunk_color = '#4E064F'
        share_zip_progess_bar_border_color = '#4E064F'
        share_zip_progess_bar_chunk_color = '#4E064F'
        history_background_color = '#ffffff'
        history_label_color = '#000000'
        settings_error_color = '#FF0000'
        if color_mode == 'dark':
            header_color = '#F2F2F2'
            title_color = '#F2F2F2'
            stop_button_color = '#C32F2F'
            new_tab_button_background = '#5F5F5F'
            new_tab_button_border = '#878787'
            new_tab_button_text_color = '#FFFFFF'
            share_zip_progess_bar_border_color = '#F2F2F2'
            history_background_color = '#191919'
            history_label_color = '#ffffff'
            settings_error_color = '#FF9999'
        return {'tab_widget': '\n                QTabBar::tab { width: 170px; height: 30px; }\n                ', 'tab_widget_new_tab_button': '\n                QPushButton {\n                    font-weight: bold;\n                    font-size: 20px;\n                }', 'settings_subtab_bar': '\n                QTabBar::tab {\n                    background: transparent;\n                }\n                QTabBar::tab:selected {\n                    border-bottom: 3px solid;\n                    border-color: #4E064F;\n                    padding: 3px\n                }', 'mode_new_tab_button': '\n                QPushButton {\n                    font-weight: bold;\n                    font-size: 30px;\n                    color: #601f61;\n                }', 'mode_header_label': '\n                QLabel {\n                    color: ' + header_color + ';\n                    font-size: 48px;\n                    margin-bottom: 16px;\n                }', 'settings_button': '\n                QPushButton {\n                    border: 0;\n                    border-radius: 0;\n                }', 'server_status_indicator_label': '\n                QLabel {\n                    font-style: italic;\n                    color: #666666;\n                    padding: 2px;\n                }', 'status_bar': '\n                QStatusBar {\n                    font-style: italic;\n                    color: #666666;\n                }\n                QStatusBar::item {\n                    border: 0px;\n                }', 'autoconnect_start_button': '\n                QPushButton {\n                    background-color: #5fa416;\n                    color: #ffffff;\n                    padding: 10px;\n                    border: 0;\n                    border-radius: 5px;\n                }', 'autoconnect_configure_button': '\n                QPushButton {\n                    padding: 9px 29px;\n                    color: #3f7fcf;\n                    text-align: left;\n                }', 'enable_autoconnect': '\n                QCheckBox {\n                    margin-top: 30px;\n                    background: #FCFCFC;\n                    color: #000000;\n                    border: 1px solid #DDDBDA;\n                    border-radius: 8px;\n                    padding: 24px 16px;\n                }\n                QCheckBox::indicator {\n                    width: 0;\n                    height: 0;\n                }', 'autoconnect_countries_combobox': '\n                QComboBox {\n                    padding: 10px;\n                    font-size: 16px;\n                    margin-left: 32px;\n                }\n                QComboBox:disabled {\n                    color: #666666;\n                }\n                ', 'autoconnect_task_label': '\n                QLabel {\n                    font-weight: bold;\n                }\n                ', 'autoconnect_failed_to_connect_label': '\n                QLabel {\n                    font-size: 18px;\n                    font-weight: bold;\n                }', 'autoconnect_bridge_setting_options': '\n                QGroupBox {\n                    border: 0;\n                    border-color: transparent;\n                    background-color: transparent;\n                    font-weight: bold;\n                    margin-top: 16px;\n                }\n                QGroupBox::title {\n                    subcontrol-origin: margin;\n                }', 'mode_settings_toggle_advanced': '\n                QPushButton {\n                    color: #3f7fcf;\n                    text-align: left;\n                }\n                ', 'mode_info_label': '\n                QLabel {\n                    font-size: 12px;\n                    color: #666666;\n                }\n                ', 'server_status_url': '\n                QLabel {\n                    background-color: #ffffff;\n                    color: #000000;\n                    padding: 10px;\n                    border: 1px solid #666666;\n                    font-size: 12px;\n                }\n                ', 'server_status_url_buttons': '\n                QPushButton {\n                    padding: 4px 8px;\n                    text-align: center;\n                }\n                ', 'server_status_button_stopped': '\n                QPushButton {\n                    background-color: #5fa416;\n                    color: #ffffff;\n                    padding: 10px 30px 10px 30px;\n                    border: 0;\n                    border-radius: 5px;\n                }', 'server_status_button_working': '\n                QPushButton {\n                    background-color: #4c8211;\n                    color: #ffffff;\n                    padding: 10px 30px 10px 30px;\n                    border: 0;\n                    border-radius: 5px;\n                    font-style: italic;\n                }', 'server_status_button_started': '\n                QPushButton {\n                    background-color: ' + stop_button_color + ';\n                    color: #ffffff;\n                    padding: 10px 30px 10px 30px;\n                    border: 0;\n                    border-radius: 5px;\n                }', 'downloads_uploads_not_empty': '\n                QWidget{\n                    background-color: ' + history_background_color + ';\n                }', 'downloads_uploads_empty': '\n                QWidget {\n                    background-color: ' + history_background_color + ';\n                    border: 1px solid #999999;\n                }\n                QWidget QLabel {\n                    background-color: none;\n                    border: 0px;\n                }\n                ', 'downloads_uploads_empty_text': '\n                QLabel {\n                    color: #999999;\n                }', 'downloads_uploads_label': '\n                QLabel {\n                    font-weight: bold;\n                    font-size 14px;\n                    text-align: center;\n                    background-color: none;\n                    border: none;\n                }', 'downloads_uploads_clear': '\n                QPushButton {\n                    color: #3f7fcf;\n                }\n                ', 'download_uploads_indicator': '\n                QLabel {\n                    color: #ffffff;\n                    background-color: #f44449;\n                    font-weight: bold;\n                    font-size: 10px;\n                    padding: 2px;\n                    border-radius: 7px;\n                    text-align: center;\n                }', 'downloads_uploads_progress_bar': '\n                QProgressBar {\n                    border: 1px solid ' + downloads_uploads_progress_bar_border_color + ';\n                    background-color: #ffffff !important;\n                    text-align: center;\n                    color: #9b9b9b;\n                    font-size: 14px;\n                }\n                QProgressBar::chunk {\n                    background-color: ' + downloads_uploads_progress_bar_chunk_color + ';\n                    width: 10px;\n                }', 'history_default_label': '\n                QLabel {\n                    color: ' + history_label_color + ';\n                }', 'history_individual_file_timestamp_label': '\n                QLabel {\n                    color: #666666;\n                }', 'history_individual_file_status_code_label_2xx': '\n                QLabel {\n                    color: #008800;\n                }', 'history_individual_file_status_code_label_4xx': '\n                QLabel {\n                    color: #cc0000;\n                }', 'tor_not_connected_label': '\n                QLabel {\n                    font-size: 16px;\n                    font-style: italic;\n                }', 'new_tab_button_image': '\n                QLabel {\n                    padding: 30px;\n                    text-align: center;\n                }\n                ', 'new_tab_button_text': '\n                QLabel {\n                    border: 1px solid ' + new_tab_button_border + ';\n                    border-radius: 4px;\n                    background-color: ' + new_tab_button_background + ';\n                    text-align: center;\n                    color: ' + new_tab_button_text_color + ';\n                }\n                ', 'new_tab_title_text': '\n                QLabel {\n                    text-align: center;\n                    color: ' + title_color + ';\n                    font-size: 25px;\n                }\n                ', 'share_delete_all_files_button': '\n                QPushButton {\n                    color: #3f7fcf;\n                }\n                ', 'share_zip_progess_bar': '\n                QProgressBar {\n                    border: 1px solid ' + share_zip_progess_bar_border_color + ';\n                    background-color: #ffffff !important;\n                    text-align: center;\n                    color: #9b9b9b;\n                }\n                QProgressBar::chunk {\n                    border: 0px;\n                    background-color: ' + share_zip_progess_bar_chunk_color + ';\n                    width: 10px;\n                }', 'share_filesize_warning': '\n                QLabel {\n                    padding: 10px 0;\n                    font-weight: bold;\n                    color: ' + title_color + ';\n                }\n                ', 'share_file_selection_drop_here_header_label': '\n                QLabel {\n                    color: ' + header_color + ';\n                    font-size: 48px;\n                }', 'share_file_selection_drop_here_label': '\n                QLabel {\n                    color: #666666;\n                }', 'share_file_selection_drop_count_label': '\n                QLabel {\n                    color: #ffffff;\n                    background-color: #f44449;\n                    font-weight: bold;\n                    padding: 5px 10px;\n                    border-radius: 10px;\n                }', 'share_file_list_drag_enter': '\n                FileList {\n                    border: 3px solid #538ad0;\n                }\n                ', 'share_file_list_drag_leave': '\n                FileList {\n                    border: none;\n                }\n                ', 'share_file_list_item_size': '\n                QLabel {\n                    color: #666666;\n                    font-size: 11px;\n                }', 'receive_file': '\n                QWidget {\n                    background-color: #ffffff;\n                }\n                ', 'receive_file_size': '\n                QLabel {\n                    color: #666666;\n                    font-size: 11px;\n                }', 'receive_message_button': '\n                QPushButton {\n                    padding: 5px 10px;\n                }', 'receive_options': '\n                QCheckBox:disabled {\n                    color: #666666;\n                }', 'tor_settings_error': '\n                QLabel {\n                    color: ' + settings_error_color + ';\n                }\n                '}

    def get_tor_paths(self):
        if False:
            i = 10
            return i + 15
        if self.common.platform == 'Linux':
            base_path = self.get_resource_path('tor')
            if base_path and os.path.isdir(base_path):
                self.common.log('GuiCommon', 'get_tor_paths', 'using paths in resources')
                tor_path = os.path.join(base_path, 'tor')
                tor_geo_ip_file_path = os.path.join(base_path, 'geoip')
                tor_geo_ipv6_file_path = os.path.join(base_path, 'geoip6')
                obfs4proxy_file_path = os.path.join(base_path, 'obfs4proxy')
                snowflake_file_path = os.path.join(base_path, 'snowflake-client')
                meek_client_file_path = os.path.join(base_path, 'meek-client')
            else:
                self.common.log('GuiCommon', 'get_tor_paths', 'using paths from PATH')
                tor_path = shutil.which('tor')
                obfs4proxy_file_path = shutil.which('obfs4proxy')
                snowflake_file_path = shutil.which('snowflake-client')
                meek_client_file_path = shutil.which('meek-client')
                prefix = os.path.dirname(os.path.dirname(tor_path))
                tor_geo_ip_file_path = os.path.join(prefix, 'share/tor/geoip')
                tor_geo_ipv6_file_path = os.path.join(prefix, 'share/tor/geoip6')
        if self.common.platform == 'Windows':
            base_path = self.get_resource_path('tor')
            tor_path = os.path.join(base_path, 'tor.exe')
            obfs4proxy_file_path = os.path.join(base_path, 'obfs4proxy.exe')
            snowflake_file_path = os.path.join(base_path, 'snowflake-client.exe')
            meek_client_file_path = os.path.join(base_path, 'meek-client.exe')
            tor_geo_ip_file_path = os.path.join(base_path, 'geoip')
            tor_geo_ipv6_file_path = os.path.join(base_path, 'geoip6')
        elif self.common.platform == 'Darwin':
            base_path = self.get_resource_path('tor')
            tor_path = os.path.join(base_path, 'tor')
            obfs4proxy_file_path = os.path.join(base_path, 'obfs4proxy')
            snowflake_file_path = os.path.join(base_path, 'snowflake-client')
            meek_client_file_path = os.path.join(base_path, 'meek-client')
            tor_geo_ip_file_path = os.path.join(base_path, 'geoip')
            tor_geo_ipv6_file_path = os.path.join(base_path, 'geoip6')
        elif self.common.platform == 'BSD':
            tor_path = '/usr/local/bin/tor'
            tor_geo_ip_file_path = '/usr/local/share/tor/geoip'
            tor_geo_ipv6_file_path = '/usr/local/share/tor/geoip6'
            obfs4proxy_file_path = '/usr/local/bin/obfs4proxy'
            meek_client_file_path = '/usr/local/bin/meek-client'
            snowflake_file_path = '/usr/local/bin/snowflake-client'
        return (tor_path, tor_geo_ip_file_path, tor_geo_ipv6_file_path, obfs4proxy_file_path, snowflake_file_path, meek_client_file_path)

    @staticmethod
    def get_resource_path(filename):
        if False:
            return 10
        '\n        Returns the absolute path of a resource\n        '
        try:
            return resource_filename('onionshare', os.path.join('resources', filename))
        except KeyError:
            return None

    @staticmethod
    def get_translated_tor_error(e):
        if False:
            i = 10
            return i + 15
        '\n        Takes an exception defined in onion.py and returns a translated error message\n        '
        if type(e) is TorErrorInvalidSetting:
            return strings._('settings_error_unknown')
        elif type(e) is TorErrorAutomatic:
            return strings._('settings_error_automatic')
        elif type(e) is TorErrorSocketPort:
            return strings._('settings_error_socket_port').format(e.args[0], e.args[1])
        elif type(e) is TorErrorSocketFile:
            return strings._('settings_error_socket_file').format(e.args[0])
        elif type(e) is TorErrorMissingPassword:
            return strings._('settings_error_missing_password')
        elif type(e) is TorErrorUnreadableCookieFile:
            return strings._('settings_error_unreadable_cookie_file')
        elif type(e) is TorErrorAuthError:
            return strings._('settings_error_auth').format(e.args[0], e.args[1])
        elif type(e) is TorErrorProtocolError:
            return strings._('error_tor_protocol_error').format(e.args[0])
        elif type(e) is BundledTorTimeout:
            return strings._('settings_error_bundled_tor_timeout')
        elif type(e) is BundledTorBroken:
            return strings._('settings_error_bundled_tor_broken').format(e.args[0])
        elif type(e) is TorTooOldEphemeral:
            return strings._('error_ephemeral_not_supported')
        elif type(e) is TorTooOldStealth:
            return strings._('error_stealth_not_supported')
        elif type(e) is PortNotAvailable:
            return strings._('error_port_not_available')
        return None

    @staticmethod
    def get_translated_web_error(e):
        if False:
            i = 10
            return i + 15
        '\n        Takes an exception defined in web.py and returns a translated error message\n        '
        if type(e) is WaitressException:
            return strings._('waitress_web_server_error')

class ToggleCheckbox(QtWidgets.QCheckBox):

    def __init__(self, text):
        if False:
            for i in range(10):
                print('nop')
        super(ToggleCheckbox, self).__init__(text)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.w = 50
        self.h = 24
        self.bg_color = '#D4D4D4'
        self.circle_color = '#BDBDBD'
        self.active_color = '#4E0D4E'
        self.inactive_color = ''

    def hitButton(self, pos):
        if False:
            for i in range(10):
                print('nop')
        return self.toggleRect.contains(pos)

    def paintEvent(self, e):
        if False:
            i = 10
            return i + 15
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(QtCore.Qt.NoPen)
        opt = QtWidgets.QStyleOptionButton()
        opt.initFrom(self)
        self.initStyleOption(opt)
        s = self.style()
        s.drawControl(QtWidgets.QStyle.CE_CheckBox, opt, painter, self)
        rect = QtCore.QRect(s.subElementRect(QtWidgets.QStyle.SE_CheckBoxContents, opt, self))
        x = rect.width() - rect.x() - self.w + 20
        y = self.height() / 2 - self.h / 2 + 16
        self.toggleRect = QtCore.QRect(x, y, self.w, self.h)
        painter.setBrush(QtGui.QColor(self.bg_color))
        painter.drawRoundedRect(x, y, self.w, self.h, self.h / 2, self.h / 2)
        if not self.isChecked():
            painter.setBrush(QtGui.QColor(self.circle_color))
            painter.drawEllipse(x, y - 3, self.h + 6, self.h + 6)
        else:
            painter.setBrush(QtGui.QColor(self.active_color))
            painter.drawEllipse(x + self.w - (self.h + 6), y - 3, self.h + 6, self.h + 6)
        painter.end()