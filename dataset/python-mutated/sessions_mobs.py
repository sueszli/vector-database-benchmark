from decouple import config
from chalicelib.utils.storage import StorageClient

def __get_mob_keys(project_id, session_id):
    if False:
        for i in range(10):
            print('nop')
    params = {'sessionId': session_id, 'projectId': project_id}
    return [config('SESSION_MOB_PATTERN_S', default='%(sessionId)s') % params, config('SESSION_MOB_PATTERN_E', default='%(sessionId)se') % params]

def __get_ios_video_keys(project_id, session_id):
    if False:
        while True:
            i = 10
    params = {'sessionId': session_id, 'projectId': project_id}
    return [config('SESSION_IOS_VIDEO_PATTERN') % params]

def __get_mob_keys_deprecated(session_id):
    if False:
        for i in range(10):
            print('nop')
    return [str(session_id), str(session_id) + 'e']

def get_urls(project_id, session_id, check_existence: bool=True):
    if False:
        return 10
    results = []
    for k in __get_mob_keys(project_id=project_id, session_id=session_id):
        if check_existence and (not StorageClient.exists(bucket=config('sessions_bucket'), key=k)):
            continue
        results.append(StorageClient.get_presigned_url_for_sharing(bucket=config('sessions_bucket'), expires_in=config('PRESIGNED_URL_EXPIRATION', cast=int, default=900), key=k))
    return results

def get_urls_depercated(session_id, check_existence: bool=True):
    if False:
        return 10
    results = []
    for k in __get_mob_keys_deprecated(session_id=session_id):
        if check_existence and (not StorageClient.exists(bucket=config('sessions_bucket'), key=k)):
            continue
        results.append(StorageClient.get_presigned_url_for_sharing(bucket=config('sessions_bucket'), expires_in=100000, key=k))
    return results

def get_ios_videos(session_id, project_id, check_existence=False):
    if False:
        while True:
            i = 10
    results = []
    for k in __get_ios_video_keys(project_id=project_id, session_id=session_id):
        if check_existence and (not StorageClient.exists(bucket=config('IOS_VIDEO_BUCKET'), key=k)):
            continue
        results.append(StorageClient.get_presigned_url_for_sharing(bucket=config('IOS_VIDEO_BUCKET'), expires_in=config('PRESIGNED_URL_EXPIRATION', cast=int, default=900), key=k))
    return results

def delete_mobs(project_id, session_ids):
    if False:
        i = 10
        return i + 15
    for session_id in session_ids:
        for k in __get_mob_keys(project_id=project_id, session_id=session_id) + __get_mob_keys_deprecated(session_id=session_id):
            StorageClient.tag_for_deletion(bucket=config('sessions_bucket'), key=k)