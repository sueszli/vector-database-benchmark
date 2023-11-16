import os
import owncloud as nextcloud

def login(user):
    if False:
        return 10
    nc = nextcloud.Client(user.nextcloud_server_address)
    nc.login(user.nextcloud_username, user.nextcloud_app_password)

    def path_to_dict(path):
        if False:
            for i in range(10):
                print('nop')
        d = {'title': os.path.basename(path), 'absolute_path': path}
        try:
            d['children'] = [path_to_dict(os.path.join(path, x.path)) for x in nc.list(path) if x.is_dir()]
        except Exception:
            pass
        return d

def list_dir(user, path):
    if False:
        print('Hello World!')
    nc = nextcloud.Client(user.nextcloud_server_address)
    nc.login(user.nextcloud_username, user.nextcloud_app_password)
    return [p.path for p in nc.list(path) if p.is_dir()]