import asyncio
import pytest
from feeluown.player import LiveLyric
from feeluown.library import LyricModel
lyric = '[by:魏积分]\n[00:38.01]大都会に\u3000僕はもう一人で\n[00:47.13]投げ捨てられた\u3000空カンのようだ\n[00:54.29]互いのすべてを\u3000知りつくすまでが\n[01:03.06]愛ならば\u3000いっそ\u3000永久（とわ）に眠ろうか\n[01:12.10]世界が終るまでは\n[01:17.10]離れる事もない\n[01:22.03]そう願っていた\n[01:26.97]幾千の夜と\n[01:30.29]戻らない時だけが\n[01:35.08]何故輝いては\n[01:41.36]やつれ切った\u3000心までも\u3000壊す\n[01:49.39]はかなき想い\n[01:54.60]このTragedy Night\n[02:01.85]\n[02:05.00]そして人は\u3000形（こたえ）を求めて\n[02:15.13]かけがえのない\u3000何かを失う\n[02:22.20]欲望だらけの\u3000街じゃ\u3000夜空の\n[02:31.19]星屑も\u3000僕らを\u3000灯せない\n[02:39.08]世界が终る前に\n[02:45.39]聞かせておくれよ\n[02:50.38]満開の花が\n[02:54.81]似合いのCatastrophe\n[02:58.65]誰もが望みながら\u3000永遠を信じない\n[03:08.71]なのに\u3000きっと\u3000明日を夢見てる\n[03:17.21]はかなき日々と\n[03:21.21]このTragedy Night\n[03:29.21]\n[03:48.06]世界が終るまでは\u3000離れる事もない\n[03:58.64]そう願っていた\u3000幾千の夜と\n[04:07.47]戻らない時だけが\u3000何故輝いては\n[04:16.12]やつれ切った\u3000心までも\u3000壊す\n[04:25.09]はかなき想い\n[04:31.41]このTragedy Night\n[04:36.31]このTragedy Night\n'
SomeoneLikeYou = '\n[00:00.000] 作词 : Adele Adkins/Dan Wilson\n[00:01.000] 作曲 : Adele Adkins/Dan Wilson\n[00:12.737]I heard\n[00:16.478]That you are settled down\n[00:21.789]That you found a girl\n[00:23.557]And you are married now\n[00:28.728]I heard\n[00:31.590]That your dreams came true\n'
SomeoneLikeYouTrans = '\n[by:baekhyun_josae]\n[00:12.737]听说\n[00:16.478]你心有所属\n[00:21.789]你遇到了她\n[00:23.557]已经步入婚姻殿堂\n[00:28.728]听说\n[00:31.590]你美梦成真\n'

class FakeLyric(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.content = lyric

@pytest.mark.asyncio
async def test_no_song_changed(app_mock):
    app_mock.library.song_get_lyric.return_value = FakeLyric()
    live_lyric = LiveLyric(app_mock)
    song = object()
    live_lyric.on_song_changed(song)
    await asyncio.sleep(0.1)
    live_lyric.on_position_changed(60)
    assert live_lyric.current_sentence == '互いのすべてを\u3000知りつくすまでが'

def test_live_lyric_with_trans(app_mock):
    if False:
        for i in range(10):
            print('nop')
    model = LyricModel(identifier=1, source='local', content=SomeoneLikeYou, trans_content=SomeoneLikeYouTrans)
    live_lyric = LiveLyric(app_mock)
    live_lyric.set_lyric(model)
    live_lyric.on_position_changed(3)
    assert live_lyric.current_sentence == ' 作曲 : Adele Adkins/Dan Wilson'
    current_line = live_lyric.current_line
    assert current_line[0] == live_lyric.current_sentence
    assert current_line[2]
    assert current_line[1] == ''
    live_lyric.on_position_changed(29)
    assert live_lyric.current_sentence == 'I heard'
    current_line = live_lyric.current_line
    assert current_line[0] == live_lyric.current_sentence
    assert current_line[1] == '听说'
    assert current_line[2]
    live_lyric.set_lyric(None)
    assert live_lyric.current_sentence == ''
    assert live_lyric.current_line[0] == ''
    assert live_lyric.current_line[2] is False