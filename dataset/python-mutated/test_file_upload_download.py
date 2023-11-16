import filecmp
from os import path
from e2b import Sandbox

def test_download():
    if False:
        print('Hello World!')
    file_name = 'video.webm'
    local_dir = 'tests/assets'
    local_path = path.join(local_dir, file_name)
    sandbox = Sandbox()
    with open(local_path, 'rb') as f:
        uploaded_file_path = sandbox.upload_file(file=f)
    file_content = sandbox.download_file(uploaded_file_path)
    with open(path.join(local_dir, 'video-downloaded.webm'), 'wb') as f:
        f.write(file_content)
    assert filecmp.cmp(local_path, path.join(local_dir, 'video-downloaded.webm'))
    sandbox.close()