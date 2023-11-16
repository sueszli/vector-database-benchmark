import json
import os
g_backup_envs = None

def getenv_or_backup(name, default=None):
    if False:
        while True:
            i = 10
    global g_backup_envs
    if g_backup_envs is None:
        backup_path = os.getenv('PADDLE_BACKUP_ENV_PATH')
        if backup_path is None:
            g_backup_envs = {}
        else:
            with open(backup_path, 'r') as f:
                g_backup_envs = json.load(f)
    value = os.getenv(name)
    if value is not None:
        return value
    else:
        return g_backup_envs.get(name, default)