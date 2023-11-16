from __future__ import absolute_import
import pwd
from typing import Union
from trashcli.lib.trash_dirs import home_trash_dir_path_from_env, home_trash_dir_path_from_home

class UserInfo:

    def __init__(self, home_trash_dir_paths, uid):
        if False:
            for i in range(10):
                print('nop')
        self.home_trash_dir_paths = home_trash_dir_paths
        self.uid = uid

class SingleUserInfoProvider:

    @staticmethod
    def get_user_info(environ, uid):
        if False:
            i = 10
            return i + 15
        return [UserInfo(home_trash_dir_path_from_env(environ), uid)]

class AllUsersInfoProvider:

    @staticmethod
    def get_user_info(_environ, _uid):
        if False:
            print('Hello World!')
        for user in pwd.getpwall():
            yield UserInfo([home_trash_dir_path_from_home(user.pw_dir)], user.pw_uid)
UserInfoProvider = Union[SingleUserInfoProvider, AllUsersInfoProvider]