from persepolis.gui.progress_ui import ProgressWindow_Ui
try:
    from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel
    from PySide6.QtCore import QCoreApplication
except:
    from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel
    from PyQt5.QtCore import QCoreApplication

class VideoFinderProgressWindow_Ui(ProgressWindow_Ui):

    def __init__(self, persepolis_setting):
        if False:
            while True:
                i = 10
        super().__init__(persepolis_setting)
        self.status_tab = QWidget()
        status_tab_verticalLayout = QVBoxLayout(self.status_tab)
        self.video_status_label = QLabel(self.status_tab)
        status_tab_verticalLayout.addWidget(self.video_status_label)
        self.audio_status_label = QLabel(self.status_tab)
        status_tab_verticalLayout.addWidget(self.audio_status_label)
        self.muxing_status_label = QLabel(self.status_tab)
        status_tab_verticalLayout.addWidget(self.muxing_status_label)
        self.progress_tabWidget.addTab(self.status_tab, '')
        self.progress_tabWidget.setCurrentIndex(2)
        self.video_status_label.setText(QCoreApplication.translate('video_finder_progress_ui_tr', '<b>Video file status: </b>'))
        self.audio_status_label.setText(QCoreApplication.translate('video_finder_progress_ui_tr', '<b>Audio file status: </b>'))
        self.muxing_status_label.setText(QCoreApplication.translate('video_finder_progress_ui_tr', '<b>Mixing status: </b>'))
        self.progress_tabWidget.setTabText(self.progress_tabWidget.indexOf(self.status_tab), QCoreApplication.translate('setting_ui_tr', 'Status'))