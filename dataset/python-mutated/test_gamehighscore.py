import pytest
from telegram import GameHighScore, User
from tests.auxil.slots import mro_slots

@pytest.fixture(scope='module')
def game_highscore():
    if False:
        i = 10
        return i + 15
    return GameHighScore(TestGameHighScoreBase.position, TestGameHighScoreBase.user, TestGameHighScoreBase.score)

class TestGameHighScoreBase:
    position = 12
    user = User(2, 'test user', False)
    score = 42

class TestGameHighScoreWithoutRequest(TestGameHighScoreBase):

    def test_slot_behaviour(self, game_highscore):
        if False:
            while True:
                i = 10
        for attr in game_highscore.__slots__:
            assert getattr(game_highscore, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(game_highscore)) == len(set(mro_slots(game_highscore))), 'same slot'

    def test_de_json(self, bot):
        if False:
            i = 10
            return i + 15
        json_dict = {'position': self.position, 'user': self.user.to_dict(), 'score': self.score}
        highscore = GameHighScore.de_json(json_dict, bot)
        assert highscore.api_kwargs == {}
        assert highscore.position == self.position
        assert highscore.user == self.user
        assert highscore.score == self.score
        assert GameHighScore.de_json(None, bot) is None

    def test_to_dict(self, game_highscore):
        if False:
            return 10
        game_highscore_dict = game_highscore.to_dict()
        assert isinstance(game_highscore_dict, dict)
        assert game_highscore_dict['position'] == game_highscore.position
        assert game_highscore_dict['user'] == game_highscore.user.to_dict()
        assert game_highscore_dict['score'] == game_highscore.score

    def test_equality(self):
        if False:
            i = 10
            return i + 15
        a = GameHighScore(1, User(2, 'test user', False), 42)
        b = GameHighScore(1, User(2, 'test user', False), 42)
        c = GameHighScore(2, User(2, 'test user', False), 42)
        d = GameHighScore(1, User(3, 'test user', False), 42)
        e = User(3, 'test user', False)
        assert a == b
        assert hash(a) == hash(b)
        assert a != c
        assert hash(a) != hash(c)
        assert a != d
        assert hash(a) != hash(d)
        assert a != e
        assert hash(a) != hash(e)