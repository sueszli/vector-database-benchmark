import logging
from fuo_netease.models import NSongModel
logging.basicConfig()
logger = logging.getLogger('feeluown')
logger.setLevel(logging.DEBUG)

def test_model_display():
    if False:
        for i in range(10):
            print('nop')
    song = NSongModel.create_by_display(identifier=254548, title='成全', artists_name='刘若英')
    assert song.album_name_display == ''
    assert song.title_display == '成全'
    print(song.url, song.title)
    assert song.album_name_display != ''

def main():
    if False:
        return 10
    test_model_display()
if __name__ == '__main__':
    main()