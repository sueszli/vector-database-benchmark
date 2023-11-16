from pathlib import Path

class Replacement:

    def __init__(self, file: Path, offset: int, length: int, replacement_text: str):
        if False:
            i = 10
            return i + 15
        ' Replacement text for file between offset and offset+length.\n\n        @param file: File to replace text in\n        @param offset: Offset in file to start text replace\n        @param length: Length of text that will be replaced. offset -> offset+length is the section of text to replace.\n        @param replacement_text: Text to insert of offset in file.\n        '
        self.file = file
        self.offset = offset
        self.length = length
        self.replacement_text = replacement_text

    def toDict(self) -> dict:
        if False:
            return 10
        return {'FilePath': self.file.as_posix(), 'Offset': self.offset, 'Length': self.length, 'ReplacementText': self.replacement_text}