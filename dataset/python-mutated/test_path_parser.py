from module.conf import PLATFORM

def test_path_to_bangumi():
    if False:
        while True:
            i = 10
    from module.downloader.path import TorrentPath
    path = 'Downloads/Bangumi/Kono Subarashii Sekai ni Shukufuku wo!/Season 2/'
    (bangumi_name, season) = TorrentPath()._path_to_bangumi(path)
    assert bangumi_name == 'Kono Subarashii Sekai ni Shukufuku wo!'
    assert season == 2