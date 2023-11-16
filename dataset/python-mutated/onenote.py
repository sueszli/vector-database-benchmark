import json
import logging
from pyOneNote.Main import process_onenote_file
from api_app.analyzers_manager.classes import FileAnalyzer
logger = logging.getLogger(__name__)

class OneNoteInfo(FileAnalyzer):

    def run(self):
        if False:
            print('Hello World!')
        with open(self.filepath, 'rb') as file:
            results = json.loads(process_onenote_file(file, '', '', True))
        return results