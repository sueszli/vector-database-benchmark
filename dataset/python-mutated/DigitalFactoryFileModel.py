from typing import List, Dict, Callable
from PyQt6.QtCore import Qt, pyqtSignal
from UM.Logger import Logger
from UM.Qt.ListModel import ListModel
from .DigitalFactoryFileResponse import DigitalFactoryFileResponse
DIGITAL_FACTORY_DISPLAY_DATETIME_FORMAT = '%d-%m-%Y %H:%M'

class DigitalFactoryFileModel(ListModel):
    FileNameRole = Qt.ItemDataRole.UserRole + 1
    FileIdRole = Qt.ItemDataRole.UserRole + 2
    FileSizeRole = Qt.ItemDataRole.UserRole + 3
    LibraryProjectIdRole = Qt.ItemDataRole.UserRole + 4
    DownloadUrlRole = Qt.ItemDataRole.UserRole + 5
    UsernameRole = Qt.ItemDataRole.UserRole + 6
    UploadedAtRole = Qt.ItemDataRole.UserRole + 7
    dfFileModelChanged = pyqtSignal()

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        super().__init__(parent)
        self.addRoleName(self.FileNameRole, 'fileName')
        self.addRoleName(self.FileIdRole, 'fileId')
        self.addRoleName(self.FileSizeRole, 'fileSize')
        self.addRoleName(self.LibraryProjectIdRole, 'libraryProjectId')
        self.addRoleName(self.DownloadUrlRole, 'downloadUrl')
        self.addRoleName(self.UsernameRole, 'username')
        self.addRoleName(self.UploadedAtRole, 'uploadedAt')
        self._files = []
        self._filters = {}

    def setFiles(self, df_files_in_project: List[DigitalFactoryFileResponse]) -> None:
        if False:
            while True:
                i = 10
        if self._files == df_files_in_project:
            return
        self.clear()
        self._files = df_files_in_project
        self._update()

    def clearFiles(self) -> None:
        if False:
            while True:
                i = 10
        self.clear()
        self._files.clear()
        self.dfFileModelChanged.emit()

    def _update(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        filtered_files_list = self.getFilteredFilesList()
        for file in filtered_files_list:
            self.appendItem({'fileName': file.file_name, 'fileId': file.file_id, 'fileSize': file.file_size, 'libraryProjectId': file.library_project_id, 'downloadUrl': file.download_url, 'username': file.username, 'uploadedAt': file.uploaded_at.strftime(DIGITAL_FACTORY_DISPLAY_DATETIME_FORMAT)})
        self.dfFileModelChanged.emit()

    def setFilters(self, filters: Dict[str, Callable]) -> None:
        if False:
            while True:
                i = 10
        '\n        Sets the filters and updates the files model to contain only the files that meet all of the filters.\n\n        :param filters: The filters to be applied\n            example:\n            {\n                "attribute_name1": function_to_be_applied_on_DigitalFactoryFileResponse_attribute1,\n                "attribute_name2": function_to_be_applied_on_DigitalFactoryFileResponse_attribute2\n            }\n        '
        self.clear()
        self._filters = filters
        self._update()

    def clearFilters(self) -> None:
        if False:
            return 10
        '\n        Clears all the model filters\n        '
        self.setFilters({})

    def getFilteredFilesList(self) -> List[DigitalFactoryFileResponse]:
        if False:
            i = 10
            return i + 15
        '\n        Lists the files that meet all the filters specified in the self._filters. This is achieved by applying each\n        filter function on the corresponding attribute for all the filters in the self._filters. If all of them are\n        true, the file is added to the filtered files list.\n        In order for this to work, the self._filters should be in the format:\n        {\n            "attribute_name": function_to_be_applied_on_the_DigitalFactoryFileResponse_attribute\n        }\n\n        :return: The list of files that meet all the specified filters\n        '
        if not self._filters:
            return self._files
        filtered_files_list = []
        for file in self._files:
            filter_results = []
            for (attribute, filter_func) in self._filters.items():
                try:
                    filter_results.append(filter_func(getattr(file, attribute)))
                except AttributeError:
                    Logger.log('w', "Attribute '{}' doesn't exist in objects of type '{}'".format(attribute, type(file)))
            all_filters_met = all(filter_results)
            if all_filters_met:
                filtered_files_list.append(file)
        return filtered_files_list