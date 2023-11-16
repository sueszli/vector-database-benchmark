from PyQt6.QtGui import QImage
from PyQt6.QtQuick import QQuickImageProvider
from PyQt6.QtCore import QSize
from UM.Application import Application
from typing import Tuple

class PrintJobPreviewImageProvider(QQuickImageProvider):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(QQuickImageProvider.ImageType.Image)

    def requestImage(self, id: str, size: QSize) -> Tuple[QImage, QSize]:
        if False:
            return 10
        'Request a new image.\n\n        :param id: id of the requested image\n        :param size: is not used defaults to QSize(15, 15)\n        :return: an tuple containing the image and size\n        '
        uuid = id[id.find('/') + 1:]
        for output_device in Application.getInstance().getOutputDeviceManager().getOutputDevices():
            if not hasattr(output_device, 'printJobs'):
                continue
            for print_job in output_device.printJobs:
                if print_job.key == uuid:
                    if print_job.getPreviewImage():
                        return (print_job.getPreviewImage(), QSize(15, 15))
                    return (QImage(), QSize(15, 15))
        return (QImage(), QSize(15, 15))