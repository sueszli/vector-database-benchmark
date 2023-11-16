import os
import os.path
from UM.Application import Application
from UM.Logger import Logger
from UM.Message import Message
from UM.FileHandler.WriteFileJob import WriteFileJob
from UM.FileHandler.FileWriter import FileWriter
from UM.Scene.Iterator.BreadthFirstIterator import BreadthFirstIterator
from UM.OutputDevice.OutputDevice import OutputDevice
from UM.OutputDevice import OutputDeviceError
from UM.i18n import i18nCatalog
catalog = i18nCatalog('cura')

class RemovableDriveOutputDevice(OutputDevice):

    def __init__(self, device_id, device_name):
        if False:
            print('Hello World!')
        super().__init__(device_id)
        self.setName(device_name)
        self.setShortDescription(catalog.i18nc("@action:button Preceded by 'Ready to'.", 'Save to Removable Drive'))
        self.setDescription(catalog.i18nc('@item:inlistbox', 'Save to Removable Drive {0}').format(device_name))
        self.setIconName('save_sd')
        self.setPriority(1)
        self._writing = False
        self._stream = None

    def requestWrite(self, nodes, file_name=None, filter_by_machine=False, file_handler=None, **kwargs):
        if False:
            print('Hello World!')
        'Request the specified nodes to be written to the removable drive.\n\n        :param nodes: A collection of scene nodes that should be written to the\n            removable drive.\n        :param file_name: :type{string} A suggestion for the file name to write to.\n            If none is provided, a file name will be made from the names of the\n        meshes.\n        :param limit_mimetypes: Should we limit the available MIME types to the\n        MIME types available to the currently active machine?\n\n        '
        filter_by_machine = True
        if self._writing:
            raise OutputDeviceError.DeviceBusyError()
        if file_handler:
            file_formats = file_handler.getSupportedFileTypesWrite()
        else:
            file_formats = Application.getInstance().getMeshFileHandler().getSupportedFileTypesWrite()
        if filter_by_machine:
            container = Application.getInstance().getGlobalContainerStack().findContainer({'file_formats': '*'})
            machine_file_formats = [file_type.strip() for file_type in container.getMetaDataEntry('file_formats').split(';')]
            format_by_mimetype = {format['mime_type']: format for format in file_formats}
            file_formats = [format_by_mimetype[mimetype] for mimetype in machine_file_formats if mimetype in format_by_mimetype]
        if len(file_formats) == 0:
            Logger.log('e', 'There are no file formats available to write with!')
            raise OutputDeviceError.WriteRequestFailedError(catalog.i18nc('@info:status', 'There are no file formats available to write with!'))
        preferred_format = file_formats[0]
        if file_handler is not None:
            writer = file_handler.getWriterByMimeType(preferred_format['mime_type'])
        else:
            writer = Application.getInstance().getMeshFileHandler().getWriterByMimeType(preferred_format['mime_type'])
        extension = preferred_format['extension']
        if file_name is None:
            file_name = self._automaticFileName(nodes)
        if extension:
            extension = '.' + extension
        file_name = os.path.join(self.getId(), file_name + extension)
        self._performWrite(file_name, preferred_format, writer, nodes)

    def _performWrite(self, file_name, preferred_format, writer, nodes):
        if False:
            for i in range(10):
                print('nop')
        'Writes the specified nodes to the removable drive. This is split from\n        requestWrite to allow interception in other plugins. See Ultimaker/Cura#10917.\n\n        :param file_name: File path to write to.\n        :param preferred_format: Preferred file format to write to.\n        :param writer: Writer for writing to the file.\n        :param nodes: A collection of scene nodes that should be written to the\n        file.\n        '
        try:
            Logger.log('d', 'Writing to %s', file_name)
            if preferred_format['mode'] == FileWriter.OutputMode.TextMode:
                self._stream = open(file_name, 'wt', buffering=1, encoding='utf-8')
            else:
                self._stream = open(file_name, 'wb', buffering=1)
            job = WriteFileJob(writer, self._stream, nodes, preferred_format['mode'])
            job.setFileName(file_name)
            job.progress.connect(self._onProgress)
            job.finished.connect(self._onFinished)
            message = Message(catalog.i18nc("@info:progress Don't translate the XML tags <filename>!", 'Saving to Removable Drive <filename>{0}</filename>').format(self.getName()), 0, False, -1, catalog.i18nc('@info:title', 'Saving'))
            message.show()
            self.writeStarted.emit(self)
            job.setMessage(message)
            self._writing = True
            job.start()
        except PermissionError as e:
            Logger.log('e', 'Permission denied when trying to write to %s: %s', file_name, str(e))
            raise OutputDeviceError.PermissionDeniedError(catalog.i18nc("@info:status Don't translate the XML tags <filename> or <message>!", 'Could not save to <filename>{0}</filename>: <message>{1}</message>').format(file_name, str(e))) from e
        except OSError as e:
            Logger.log('e', 'Operating system would not let us write to %s: %s', file_name, str(e))
            raise OutputDeviceError.WriteRequestFailedError(catalog.i18nc("@info:status Don't translate the XML tags <filename> or <message>!", 'Could not save to <filename>{0}</filename>: <message>{1}</message>').format(file_name, str(e))) from e

    def _automaticFileName(self, nodes):
        if False:
            while True:
                i = 10
        'Generate a file name automatically for the specified nodes to be saved in.\n\n        The name generated will be the name of one of the nodes. Which node that\n        is can not be guaranteed.\n\n        :param nodes: A collection of nodes for which to generate a file name.\n        '
        for root in nodes:
            for child in BreadthFirstIterator(root):
                if child.getMeshData():
                    name = child.getName()
                    if name:
                        return name
        raise OutputDeviceError.WriteRequestFailedError(catalog.i18nc("@info:status Don't translate the tag {device}!", 'Could not find a file name when trying to write to {device}.').format(device=self.getName()))

    def _onProgress(self, job, progress):
        if False:
            while True:
                i = 10
        self.writeProgress.emit(self, progress)

    def _onFinished(self, job):
        if False:
            print('Hello World!')
        if self._stream:
            error = job.getError()
            try:
                self._stream.close()
            except Exception as e:
                if not error:
                    error = e
            self._stream = None
            self._writing = False
            self.writeFinished.emit(self)
            if not error:
                message = Message(catalog.i18nc('@info:status', 'Saved to Removable Drive {0} as {1}').format(self.getName(), os.path.basename(job.getFileName())), title=catalog.i18nc('@info:title', 'File Saved'), message_type=Message.MessageType.POSITIVE)
                message.addAction('eject', catalog.i18nc('@action:button', 'Eject'), 'eject', catalog.i18nc('@action', 'Eject removable device {0}').format(self.getName()))
                message.actionTriggered.connect(self._onActionTriggered)
                message.show()
                self.writeSuccess.emit(self)
            else:
                try:
                    os.remove(job.getFileName())
                except Exception as e:
                    Logger.logException('e', 'Exception when trying to remove incomplete exported file %s', str(job.getFileName()))
                message = Message(catalog.i18nc('@info:status', 'Could not save to removable drive {0}: {1}').format(self.getName(), str(job.getError())), title=catalog.i18nc('@info:title', 'Error'), message_type=Message.MessageType.ERROR)
                message.show()
                self.writeError.emit(self)

    def _onActionTriggered(self, message, action):
        if False:
            print('Hello World!')
        if action == 'eject':
            if Application.getInstance().getOutputDeviceManager().getOutputDevicePlugin('RemovableDriveOutputDevice').ejectDevice(self):
                message.hide()
                eject_message = Message(catalog.i18nc('@info:status', 'Ejected {0}. You can now safely remove the drive.').format(self.getName()), title=catalog.i18nc('@info:title', 'Safely Remove Hardware'))
            else:
                eject_message = Message(catalog.i18nc('@info:status', 'Failed to eject {0}. Another program may be using the drive.').format(self.getName()), title=catalog.i18nc('@info:title', 'Warning'), message_type=Message.MessageType.ERROR)
            eject_message.show()