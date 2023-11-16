import base64
from UM.Logger import Logger
from cura.Snapshot import Snapshot
from PyQt6.QtCore import QByteArray, QIODevice, QBuffer
from ..Script import Script

class CreateThumbnail(Script):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()

    def _createSnapshot(self, width, height):
        if False:
            while True:
                i = 10
        Logger.log('d', 'Creating thumbnail image...')
        try:
            return Snapshot.snapshot(width, height)
        except Exception:
            Logger.logException('w', 'Failed to create snapshot image')

    def _encodeSnapshot(self, snapshot):
        if False:
            for i in range(10):
                print('nop')
        Logger.log('d', 'Encoding thumbnail image...')
        try:
            thumbnail_buffer = QBuffer()
            thumbnail_buffer.open(QBuffer.OpenModeFlag.ReadWrite)
            thumbnail_image = snapshot
            thumbnail_image.save(thumbnail_buffer, 'PNG')
            base64_bytes = base64.b64encode(thumbnail_buffer.data())
            base64_message = base64_bytes.decode('ascii')
            thumbnail_buffer.close()
            return base64_message
        except Exception:
            Logger.logException('w', 'Failed to encode snapshot image')

    def _convertSnapshotToGcode(self, encoded_snapshot, width, height, chunk_size=78):
        if False:
            i = 10
            return i + 15
        gcode = []
        encoded_snapshot_length = len(encoded_snapshot)
        gcode.append(';')
        gcode.append('; thumbnail begin {}x{} {}'.format(width, height, encoded_snapshot_length))
        chunks = ['; {}'.format(encoded_snapshot[i:i + chunk_size]) for i in range(0, len(encoded_snapshot), chunk_size)]
        gcode.extend(chunks)
        gcode.append('; thumbnail end')
        gcode.append(';')
        gcode.append('')
        return gcode

    def getSettingDataString(self):
        if False:
            i = 10
            return i + 15
        return '{\n            "name": "Create Thumbnail",\n            "key": "CreateThumbnail",\n            "metadata": {},\n            "version": 2,\n            "settings":\n            {\n                "width":\n                {\n                    "label": "Width",\n                    "description": "Width of the generated thumbnail",\n                    "unit": "px",\n                    "type": "int",\n                    "default_value": 32,\n                    "minimum_value": "0",\n                    "minimum_value_warning": "12",\n                    "maximum_value_warning": "800"\n                },\n                "height":\n                {\n                    "label": "Height",\n                    "description": "Height of the generated thumbnail",\n                    "unit": "px",\n                    "type": "int",\n                    "default_value": 32,\n                    "minimum_value": "0",\n                    "minimum_value_warning": "12",\n                    "maximum_value_warning": "600"\n                }\n            }\n        }'

    def execute(self, data):
        if False:
            return 10
        width = self.getSettingValueByKey('width')
        height = self.getSettingValueByKey('height')
        snapshot = self._createSnapshot(width, height)
        if snapshot:
            encoded_snapshot = self._encodeSnapshot(snapshot)
            snapshot_gcode = self._convertSnapshotToGcode(encoded_snapshot, width, height)
            for layer in data:
                layer_index = data.index(layer)
                lines = data[layer_index].split('\n')
                for line in lines:
                    if line.startswith(';Generated with Cura'):
                        line_index = lines.index(line)
                        insert_index = line_index + 1
                        lines[insert_index:insert_index] = snapshot_gcode
                        break
                final_lines = '\n'.join(lines)
                data[layer_index] = final_lines
        return data