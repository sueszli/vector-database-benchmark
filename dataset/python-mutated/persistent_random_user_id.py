"""
This class is responsible for getting/setting an anonymous (GDPR-compliant) ID
"""
import sysconfig
import typing
from pathlib import Path
from borb.license.uuid import UUID

class PersistentRandomUserID:
    """
    This class is responsible for getting/setting an anonymous (GDPR-compliant) ID
    """
    USER_ID_FILE_NAME: str = 'anonymous_user_id'
    USER_ID: typing.Optional[str] = None

    @staticmethod
    def _get_borb_installation_dir() -> typing.Optional[Path]:
        if False:
            return 10
        for path_name in sysconfig.get_path_names():
            installation_path: Path = Path(sysconfig.get_path(path_name))
            if not installation_path.exists():
                continue
            borb_dir: Path = installation_path / 'borb'
            if borb_dir.exists():
                return borb_dir
        return None

    @staticmethod
    def _get_user_id_file_from_borb_dir() -> typing.Optional[Path]:
        if False:
            return 10
        installation_dir: typing.Optional[Path] = PersistentRandomUserID._get_borb_installation_dir()
        if installation_dir is None:
            return None
        user_id_file: Path = installation_dir / PersistentRandomUserID.USER_ID_FILE_NAME
        if user_id_file.exists():
            return user_id_file
        return None

    @staticmethod
    def get() -> typing.Optional[str]:
        if False:
            print('Hello World!')
        '\n        This function (creates and then) returns an anonymous user ID.\n        This ID is stored in a file in the borb installation directory to ensure consistency between calls.\n        :return:    an anonymous user ID\n        '
        installation_dir: typing.Optional[Path] = PersistentRandomUserID._get_borb_installation_dir()
        user_id_file: typing.Optional[Path] = PersistentRandomUserID._get_user_id_file_from_borb_dir()
        if installation_dir is not None and installation_dir.exists() and (user_id_file is None or not user_id_file.exists()):
            try:
                new_uuid: str = UUID.get()
                with open(installation_dir / PersistentRandomUserID.USER_ID_FILE_NAME, 'w') as fh:
                    fh.write(new_uuid)
                return new_uuid
            except:
                pass
        if user_id_file is not None and user_id_file.exists():
            prev_uuid: typing.Optional[str] = None
            try:
                with open(user_id_file, 'r') as fh:
                    prev_uuid = fh.read()
            except:
                pass
            return prev_uuid
        return None