import time
import math
import sys
import os
from typing import List, Optional
from PyQt5.QtMultimedia import QCameraInfo, QCamera, QCameraViewfinderSettings
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QCheckBox, QPushButton, QLabel, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QSize, QRect, Qt, pyqtSignal, PYQT_VERSION
from electrum.simple_config import SimpleConfig
from electrum.i18n import _
from electrum.qrreader import get_qr_reader, QrCodeResult, MissingQrDetectionLib
from electrum.logging import Logger
from electrum.gui.qt.util import MessageBoxMixin, FixedAspectRatioLayout, ImageGraphicsEffect
from .video_widget import QrReaderVideoWidget
from .video_overlay import QrReaderVideoOverlay
from .video_surface import QrReaderVideoSurface
from .crop_blur_effect import QrReaderCropBlurEffect
from .validator import AbstractQrReaderValidator, QrReaderValidatorCounted, QrReaderValidatorResult

class CameraError(RuntimeError):
    """ Base class of the camera-related error conditions. """

class NoCamerasFound(CameraError):
    """ Raised by start_scan if no usable cameras were found. Interested
    code can catch this specific exception."""

class NoCameraResolutionsFound(CameraError):
    """ Raised internally if no usable camera resolutions were found. """

class QrReaderCameraDialog(Logger, MessageBoxMixin, QDialog):
    """
    Dialog for reading QR codes from a camera
    """
    SCAN_SIZE: int = 512
    qr_finished = pyqtSignal(bool, str, object)

    def __init__(self, parent: Optional[QWidget], *, config: SimpleConfig):
        if False:
            print('Hello World!')
        ' Note: make sure parent is a "top_level_window()" as per\n        MessageBoxMixin API else bad things can happen on macOS. '
        QDialog.__init__(self, parent=parent)
        Logger.__init__(self)
        self.validator: AbstractQrReaderValidator = None
        self.frame_id: int = 0
        self.qr_crop: QRect = None
        self.qrreader_res: List[QrCodeResult] = []
        self.validator_res: QrReaderValidatorResult = None
        self.last_stats_time: float = 0.0
        self.frame_counter: int = 0
        self.qr_frame_counter: int = 0
        self.last_qr_scan_ts: float = 0.0
        self.camera: QCamera = None
        self._error_message: str = None
        self._ok_done: bool = False
        self.camera_sc_conn = None
        self.resolution: QSize = None
        self.config = config
        self.qrreader = get_qr_reader()
        flags = self.windowFlags()
        flags = flags | Qt.WindowMaximizeButtonHint
        self.setWindowFlags(flags)
        self.setWindowTitle(_('Scan QR Code'))
        self.setWindowModality(Qt.WindowModal if parent else Qt.ApplicationModal)
        self.video_widget = QrReaderVideoWidget()
        self.video_overlay = QrReaderVideoOverlay()
        self.video_layout = FixedAspectRatioLayout()
        self.video_layout.addWidget(self.video_widget)
        self.video_layout.addWidget(self.video_overlay)
        vbox = QVBoxLayout()
        self.setLayout(vbox)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.addLayout(self.video_layout)
        self.lowres_label = QLabel(_('Note: This camera generates frames of relatively low resolution; QR scanning accuracy may be affected'))
        self.lowres_label.setWordWrap(True)
        self.lowres_label.setAlignment(Qt.AlignVCenter | Qt.AlignHCenter)
        vbox.addWidget(self.lowres_label)
        self.lowres_label.setHidden(True)
        controls_layout = QHBoxLayout()
        controls_layout.addStretch(2)
        controls_layout.setContentsMargins(10, 10, 10, 10)
        controls_layout.setSpacing(10)
        vbox.addLayout(controls_layout)
        self.flip_x = QCheckBox()
        self.flip_x.setText(_('&Flip horizontally'))
        self.flip_x.setChecked(self.config.QR_READER_FLIP_X)
        self.flip_x.stateChanged.connect(self._on_flip_x_changed)
        controls_layout.addWidget(self.flip_x)
        close_but = QPushButton(_('&Close'))
        close_but.clicked.connect(self.reject)
        controls_layout.addWidget(close_but)
        self.video_surface = QrReaderVideoSurface(self)
        self.video_surface.frame_available.connect(self._on_frame_available)
        self.crop_blur_effect = QrReaderCropBlurEffect(self)
        self.image_effect = ImageGraphicsEffect(self, self.crop_blur_effect)
        self.finished.connect(self._boilerplate_cleanup, Qt.QueuedConnection)
        self.finished.connect(self._on_finished, Qt.QueuedConnection)

    def _on_flip_x_changed(self, _state: int):
        if False:
            i = 10
            return i + 15
        self.config.QR_READER_FLIP_X = self.flip_x.isChecked()

    def _get_resolution(self, resolutions: List[QSize], min_size: int) -> QSize:
        if False:
            return 10
        '\n        Given a list of resolutions that the camera supports this function picks the\n        lowest resolution that is at least min_size in both width and height.\n        If no resolution is found, NoCameraResolutionsFound is raised.\n        '

        def res_list_to_str(res_list: List[QSize]) -> str:
            if False:
                while True:
                    i = 10
            return ', '.join(['{}x{}'.format(r.width(), r.height()) for r in res_list])

        def check_res(res: QSize):
            if False:
                return 10
            return res.width() >= min_size and res.height() >= min_size
        self.logger.info('searching for at least {0}x{0}'.format(min_size))
        format_str = 'camera resolutions: {}'
        self.logger.info(format_str.format(res_list_to_str(resolutions)))
        candidate_resolutions = []
        ideal_resolutions = [r for r in resolutions if check_res(r)]
        less_than_ideal_resolutions = [r for r in resolutions if r not in ideal_resolutions]
        format_str = 'ideal resolutions: {}, less-than-ideal resolutions: {}'
        self.logger.info(format_str.format(res_list_to_str(ideal_resolutions), res_list_to_str(less_than_ideal_resolutions)))
        if not ideal_resolutions and (not less_than_ideal_resolutions):
            raise NoCameraResolutionsFound(_('Cannot start QR scanner, no usable camera resolution found.') + self._linux_pyqt5bug_msg())
        if not ideal_resolutions:
            self.logger.warning('No ideal resolutions found, falling back to less-than-ideal resolutions -- QR recognition may fail!')
            candidate_resolutions = less_than_ideal_resolutions
            is_ideal = False
        else:
            candidate_resolutions = ideal_resolutions
            is_ideal = True
        resolution = sorted(candidate_resolutions, key=lambda r: r.width() * r.height(), reverse=not is_ideal)[0]
        format_str = 'chosen resolution is {}x{}'
        self.logger.info(format_str.format(resolution.width(), resolution.height()))
        return (resolution, is_ideal)

    @staticmethod
    def _get_crop(resolution: QSize, scan_size: int) -> QRect:
        if False:
            return 10
        '\n        Returns a QRect that is scan_size x scan_size in the middle of the resolution\n        '
        scan_pos_x = (resolution.width() - scan_size) // 2
        scan_pos_y = (resolution.height() - scan_size) // 2
        return QRect(scan_pos_x, scan_pos_y, scan_size, scan_size)

    @staticmethod
    def _linux_pyqt5bug_msg():
        if False:
            for i in range(10):
                print('nop')
        ' Returns a string that may be appended to an exception error message\n        only if on Linux and PyQt5 < 5.12.2, otherwise returns an empty string. '
        if sys.platform == 'linux' and PYQT_VERSION < 330754 and (not os.environ.get('APPIMAGE')):
            return '\n\n' + _('If you indeed do have a usable camera connected, then this error may be caused by bugs in previous PyQt5 versions on Linux. Try installing the latest PyQt5:') + '\n\n' + 'python3 -m pip install --user -I pyqt5'
        return ''

    def start_scan(self, device: str=''):
        if False:
            i = 10
            return i + 15
        "\n        Scans a QR code from the given camera device.\n        If no QR code is found the returned string will be empty.\n        If the camera is not found or can't be opened NoCamerasFound will be raised.\n        "
        self.validator = QrReaderValidatorCounted()
        self.validator.strong_count = 5
        device_info = None
        for camera in QCameraInfo.availableCameras():
            if camera.deviceName() == device:
                device_info = camera
                break
        if not device_info:
            self.logger.info('Failed to open selected camera, trying to use default camera')
            device_info = QCameraInfo.defaultCamera()
        if not device_info or device_info.isNull():
            raise NoCamerasFound(_('Cannot start QR scanner, no usable camera found.') + self._linux_pyqt5bug_msg())
        self._init_stats()
        self.qrreader_res = []
        self.validator_res = None
        self._ok_done = False
        self._error_message = None
        if self.camera:
            self.logger.info('Warning: start_scan already called for this instance.')
        self.camera = QCamera(device_info)
        self.camera.setViewfinder(self.video_surface)
        self.camera.setCaptureMode(QCamera.CaptureViewfinder)
        self.camera_sc_conn = self.camera.statusChanged.connect(self._on_camera_status_changed, Qt.QueuedConnection)
        self.camera.error.connect(self._on_camera_error)
        self.camera.load()
    _camera_status_names = {QCamera.UnavailableStatus: _('unavailable'), QCamera.UnloadedStatus: _('unloaded'), QCamera.UnloadingStatus: _('unloading'), QCamera.LoadingStatus: _('loading'), QCamera.LoadedStatus: _('loaded'), QCamera.StandbyStatus: _('standby'), QCamera.StartingStatus: _('starting'), QCamera.StoppingStatus: _('stopping'), QCamera.ActiveStatus: _('active')}

    def _get_camera_status_name(self, status: QCamera.Status):
        if False:
            return 10
        return self._camera_status_names.get(status, _('unknown'))

    def _set_resolution(self, resolution: QSize):
        if False:
            print('Hello World!')
        self.resolution = resolution
        self.qr_crop = self._get_crop(resolution, self.SCAN_SIZE)
        self.resize(720, 540)
        self.video_overlay.set_crop(self.qr_crop)
        self.video_overlay.set_resolution(resolution)
        self.video_layout.set_aspect_ratio(resolution.width() / resolution.height())
        self.crop_blur_effect.setCrop(self.qr_crop)

    def _on_camera_status_changed(self, status: QCamera.Status):
        if False:
            while True:
                i = 10
        if self._ok_done:
            return
        self.logger.info('camera status changed to {}'.format(self._get_camera_status_name(status)))
        if status == QCamera.LoadedStatus:
            camera_resolutions = self.camera.supportedViewfinderResolutions()
            try:
                (resolution, was_ideal) = self._get_resolution(camera_resolutions, self.SCAN_SIZE)
            except RuntimeError as e:
                self._error_message = str(e)
                self.reject()
                return
            self._set_resolution(resolution)
            viewfinder_settings = QCameraViewfinderSettings()
            viewfinder_settings.setResolution(resolution)
            self.camera.setViewfinderSettings(viewfinder_settings)
            self.frame_id = 0
            self.camera.start()
            self.lowres_label.setVisible(not was_ideal)
        elif status == QCamera.UnloadedStatus or status == QCamera.UnavailableStatus:
            self._error_message = _('Cannot start QR scanner, camera is unavailable.')
            self.reject()
        elif status == QCamera.ActiveStatus:
            self.open()
    CameraErrorStrings = {QCamera.NoError: 'No Error', QCamera.CameraError: 'Camera Error', QCamera.InvalidRequestError: 'Invalid Request Error', QCamera.ServiceMissingError: 'Service Missing Error', QCamera.NotSupportedFeatureError: 'Unsupported Feature Error'}

    def _on_camera_error(self, errorCode):
        if False:
            print('Hello World!')
        errStr = self.CameraErrorStrings.get(errorCode, 'Unknown Error')
        self.logger.info(f'QCamera error: {errStr}')

    def accept(self):
        if False:
            print('Hello World!')
        self._ok_done = True
        super().accept()

    def reject(self):
        if False:
            while True:
                i = 10
        self._ok_done = True
        super().reject()

    def _boilerplate_cleanup(self):
        if False:
            for i in range(10):
                print('nop')
        self._close_camera()
        if self.isVisible():
            self.close()

    def _close_camera(self):
        if False:
            for i in range(10):
                print('nop')
        if self.camera:
            self.camera.setViewfinder(None)
            if self.camera_sc_conn:
                self.camera.statusChanged.disconnect(self.camera_sc_conn)
                self.camera_sc_conn = None
            self.camera.unload()
            self.camera = None

    def _on_finished(self, code):
        if False:
            return 10
        res = code == QDialog.Accepted and self.validator_res and self.validator_res.accepted and self.validator_res.simple_result or ''
        self.validator = None
        self.logger.info(f'closed {res}')
        self.qr_finished.emit(code == QDialog.Accepted, self._error_message, res)

    def _on_frame_available(self, frame: QImage):
        if False:
            for i in range(10):
                print('nop')
        if self._ok_done:
            return
        self.frame_id += 1
        if frame.size() != self.resolution:
            self.logger.info('Getting video data at {}x{} instead of the requested {}x{}, switching resolution.'.format(frame.size().width(), frame.size().height(), self.resolution.width(), self.resolution.height()))
            self._set_resolution(frame.size())
        flip_x = self.flip_x.isChecked()
        qr_scanned = time.time() - self.last_qr_scan_ts >= self.qrreader.interval()
        if qr_scanned:
            self.last_qr_scan_ts = time.time()
            frame_cropped = frame.copy(self.qr_crop)
            frame_y800 = frame_cropped.convertToFormat(QImage.Format_Grayscale8)
            self.qrreader_res = self.qrreader.read_qr_code(frame_y800.constBits().__int__(), frame_y800.byteCount(), frame_y800.bytesPerLine(), frame_y800.width(), frame_y800.height(), self.frame_id)
            self.validator_res = self.validator.validate_results(self.qrreader_res)
            self.video_overlay.set_results(self.qrreader_res, flip_x, self.validator_res)
            if self.validator_res.accepted:
                self.accept()
                return
        if self.image_effect:
            frame = self.image_effect.apply(frame)
        if flip_x:
            frame = frame.mirrored(True, False)
        self.video_widget.setPixmap(QPixmap.fromImage(frame))
        self._update_stats(qr_scanned)

    def _init_stats(self):
        if False:
            i = 10
            return i + 15
        self.last_stats_time = time.perf_counter()
        self.frame_counter = 0
        self.qr_frame_counter = 0

    def _update_stats(self, qr_scanned):
        if False:
            for i in range(10):
                print('nop')
        self.frame_counter += 1
        if qr_scanned:
            self.qr_frame_counter += 1
        now = time.perf_counter()
        last_stats_delta = now - self.last_stats_time
        if last_stats_delta > 1.0:
            fps = self.frame_counter / last_stats_delta
            qr_fps = self.qr_frame_counter / last_stats_delta
            if self.validator is not None:
                self.validator.strong_count = math.ceil(qr_fps / 3)
            stats_format = 'running at {} FPS, scanner at {} FPS'
            self.logger.info(stats_format.format(fps, qr_fps))
            self.frame_counter = 0
            self.qr_frame_counter = 0
            self.last_stats_time = now