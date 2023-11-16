import os
from typing import List
from zerver.lib.storage import static_path

def get_available_notification_sounds() -> List[str]:
    if False:
        print('Hello World!')
    notification_sounds_path = static_path('audio/notification_sounds')
    available_notification_sounds = []
    for file_name in os.listdir(notification_sounds_path):
        (root, ext) = os.path.splitext(file_name)
        if '.' in root:
            continue
        if ext == '.ogg':
            available_notification_sounds.append(root)
    return sorted(available_notification_sounds)