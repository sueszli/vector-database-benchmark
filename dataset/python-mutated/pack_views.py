import codecs
import mimetypes
import os
import six
from wsgiref.handlers import format_date_time
from st2api.controllers.v1.packs import BasePacksController
from st2common.exceptions.db import StackStormDBObjectNotFoundError
from st2common import log as logging
from st2common.models.api.pack import PackAPI
from st2common.persistence.pack import Pack
from st2common.content.utils import get_pack_file_abs_path
from st2common.rbac.types import PermissionType
from st2common.rbac.backends import get_rbac_backend
from st2common.router import abort
from st2common.router import Response
http_client = six.moves.http_client
__all__ = ['FilesController', 'FileController']
http_client = six.moves.http_client
LOG = logging.getLogger(__name__)
BOM_LEN = len(codecs.BOM_UTF8)
MAX_FILE_SIZE = 500 * 1000
WHITELISTED_FILE_PATHS = ['icon.png']

class BaseFileController(BasePacksController):
    model = PackAPI
    access = Pack
    supported_filters = {}
    query_options = {}

    def get_all(self):
        if False:
            i = 10
            return i + 15
        return abort(404)

    def _get_file_size(self, file_path):
        if False:
            return 10
        return self._get_file_stats(file_path=file_path)[0]

    def _get_file_stats(self, file_path):
        if False:
            return 10
        try:
            file_stats = os.stat(file_path)
        except OSError:
            return (None, None)
        return (file_stats.st_size, file_stats.st_mtime)

    def _get_file_content(self, file_path):
        if False:
            return 10
        with codecs.open(file_path, 'rb') as fp:
            content = fp.read()
        return content

    def _process_file_content(self, content):
        if False:
            return 10
        '\n        This method processes the file content and removes unicode BOM character if one is present.\n\n        Note: If we don\'t do that, files view explodes with "UnicodeDecodeError: ... invalid start\n        byte" because the json.dump doesn\'t know how to handle BOM character.\n        '
        if content.startswith(codecs.BOM_UTF8):
            content = content[BOM_LEN:]
        return content

class FilesController(BaseFileController):
    """
    Controller which allows user to retrieve content of all the files inside the pack.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super(FilesController, self).__init__()
        self.get_one_db_method = self._get_by_ref_or_id

    def get_one(self, ref_or_id, requester_user):
        if False:
            while True:
                i = 10
        '\n        Outputs the content of all the files inside the pack.\n\n        Handles requests:\n            GET /packs/views/files/<pack_ref_or_id>\n        '
        pack_db = self._get_by_ref_or_id(ref_or_id=ref_or_id)
        rbac_utils = get_rbac_backend().get_utils_class()
        rbac_utils.assert_user_has_resource_db_permission(user_db=requester_user, resource_db=pack_db, permission_type=PermissionType.PACK_VIEW)
        if not pack_db:
            msg = 'Pack with ref_or_id "%s" does not exist' % ref_or_id
            raise StackStormDBObjectNotFoundError(msg)
        pack_ref = pack_db.ref
        pack_files = pack_db.files
        result = []
        for file_path in pack_files:
            normalized_file_path = get_pack_file_abs_path(pack_ref=pack_ref, file_path=file_path)
            if not normalized_file_path or not os.path.isfile(normalized_file_path):
                continue
            file_size = self._get_file_size(file_path=normalized_file_path)
            if file_size is not None and file_size > MAX_FILE_SIZE:
                LOG.debug('Skipping file "%s" which size exceeds max file size (%s bytes)' % (normalized_file_path, MAX_FILE_SIZE))
                continue
            content = self._get_file_content(file_path=normalized_file_path)
            include_file = self._include_file(file_path=file_path, content=content)
            if not include_file:
                LOG.debug('Skipping binary file "%s"' % normalized_file_path)
                continue
            item = {'file_path': file_path, 'content': content}
            result.append(item)
        return result

    def _include_file(self, file_path, content):
        if False:
            print('Hello World!')
        '\n        Method which returns True if the following file content should be included in the response.\n\n        Right now we exclude any file with UTF8 BOM character in it - those are most likely binary\n        files such as icon, etc.\n        '
        if codecs.BOM_UTF8 in content[:1024]:
            return False
        if b'\x00' in content[:1024]:
            return False
        return True

class FileController(BaseFileController):
    """
    Controller which allows user to retrieve content of a specific file in a pack.
    """

    def get_one(self, ref_or_id, file_path, requester_user, if_none_match=None, if_modified_since=None):
        if False:
            return 10
        '\n        Outputs the content of a specific file in a pack.\n\n        Handles requests:\n            GET /packs/views/file/<pack_ref_or_id>/<file path>\n        '
        pack_db = self._get_by_ref_or_id(ref_or_id=ref_or_id)
        if not pack_db:
            msg = 'Pack with ref_or_id "%s" does not exist' % ref_or_id
            raise StackStormDBObjectNotFoundError(msg)
        if not file_path:
            raise ValueError('Missing file path')
        pack_ref = pack_db.ref
        permission_type = PermissionType.PACK_VIEW
        if file_path not in WHITELISTED_FILE_PATHS:
            rbac_utils = get_rbac_backend().get_utils_class()
            rbac_utils.assert_user_has_resource_db_permission(user_db=requester_user, resource_db=pack_db, permission_type=permission_type)
        normalized_file_path = get_pack_file_abs_path(pack_ref=pack_ref, file_path=file_path)
        if not normalized_file_path or not os.path.isfile(normalized_file_path):
            raise StackStormDBObjectNotFoundError('File "%s" not found' % file_path)
        (file_size, file_mtime) = self._get_file_stats(file_path=normalized_file_path)
        response = Response()
        if not self._is_file_changed(file_mtime, if_none_match=if_none_match, if_modified_since=if_modified_since):
            response.status = http_client.NOT_MODIFIED
        else:
            if file_size is not None and file_size > MAX_FILE_SIZE:
                msg = 'File %s exceeds maximum allowed file size (%s bytes)' % (file_path, MAX_FILE_SIZE)
                raise ValueError(msg)
            content_type = mimetypes.guess_type(normalized_file_path)[0] or 'application/octet-stream'
            response.headers['Content-Type'] = content_type
            response.body = self._get_file_content(file_path=normalized_file_path)
        response.headers['Last-Modified'] = format_date_time(file_mtime)
        response.headers['ETag'] = repr(file_mtime)
        return response

    def _is_file_changed(self, file_mtime, if_none_match=None, if_modified_since=None):
        if False:
            print('Hello World!')
        if if_none_match:
            return repr(file_mtime) != if_none_match
        if if_modified_since:
            return if_modified_since != format_date_time(file_mtime)
        return True

class PackViewsController(object):
    files = FilesController()
    file = FileController()