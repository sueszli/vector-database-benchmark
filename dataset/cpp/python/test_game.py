#!/usr/bin/env python
#
# A library that provides a Python interface to the Telegram Bot API
# Copyright (C) 2015-2023
# Leandro Toledo de Souza <devs@python-telegram-bot.org>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser Public License for more details.
#
# You should have received a copy of the GNU Lesser Public License
# along with this program.  If not, see [http://www.gnu.org/licenses/].

import pytest

from telegram import Animation, Game, MessageEntity, PhotoSize
from tests.auxil.slots import mro_slots


@pytest.fixture(scope="module")
def game():
    game = Game(
        TestGameBase.title,
        TestGameBase.description,
        TestGameBase.photo,
        text=TestGameBase.text,
        text_entities=TestGameBase.text_entities,
        animation=TestGameBase.animation,
    )
    game._unfreeze()
    return game


class TestGameBase:
    title = "Python-telegram-bot Test Game"
    description = "description"
    photo = [PhotoSize("Blah", "ElseBlah", 640, 360, file_size=0)]
    text = (
        b"\\U0001f469\\u200d\\U0001f469\\u200d\\U0001f467"
        b"\\u200d\\U0001f467\\U0001f431http://google.com"
    ).decode("unicode-escape")
    text_entities = [MessageEntity(13, 17, MessageEntity.URL)]
    animation = Animation("blah", "unique_id", 320, 180, 1)


class TestGameWithoutRequest(TestGameBase):
    def test_slot_behaviour(self, game):
        for attr in game.__slots__:
            assert getattr(game, attr, "err") != "err", f"got extra slot '{attr}'"
        assert len(mro_slots(game)) == len(set(mro_slots(game))), "duplicate slot"

    def test_de_json_required(self, bot):
        json_dict = {
            "title": self.title,
            "description": self.description,
            "photo": [self.photo[0].to_dict()],
        }
        game = Game.de_json(json_dict, bot)
        assert game.api_kwargs == {}

        assert game.title == self.title
        assert game.description == self.description
        assert game.photo == tuple(self.photo)

    def test_de_json_all(self, bot):
        json_dict = {
            "title": self.title,
            "description": self.description,
            "photo": [self.photo[0].to_dict()],
            "text": self.text,
            "text_entities": [self.text_entities[0].to_dict()],
            "animation": self.animation.to_dict(),
        }
        game = Game.de_json(json_dict, bot)
        assert game.api_kwargs == {}

        assert game.title == self.title
        assert game.description == self.description
        assert game.photo == tuple(self.photo)
        assert game.text == self.text
        assert game.text_entities == tuple(self.text_entities)
        assert game.animation == self.animation

    def test_to_dict(self, game):
        game_dict = game.to_dict()

        assert isinstance(game_dict, dict)
        assert game_dict["title"] == game.title
        assert game_dict["description"] == game.description
        assert game_dict["photo"] == [game.photo[0].to_dict()]
        assert game_dict["text"] == game.text
        assert game_dict["text_entities"] == [game.text_entities[0].to_dict()]
        assert game_dict["animation"] == game.animation.to_dict()

    def test_equality(self):
        a = Game("title", "description", [PhotoSize("Blah", "unique_id", 640, 360, file_size=0)])
        b = Game(
            "title",
            "description",
            [PhotoSize("Blah", "unique_id", 640, 360, file_size=0)],
            text="Here is a text",
        )
        c = Game(
            "eltit",
            "description",
            [PhotoSize("Blah", "unique_id", 640, 360, file_size=0)],
            animation=Animation("blah", "unique_id", 320, 180, 1),
        )
        d = Animation("blah", "unique_id", 320, 180, 1)

        assert a == b
        assert hash(a) == hash(b)

        assert a != c
        assert hash(a) != hash(c)

        assert a != d
        assert hash(a) != hash(d)

    def test_parse_entity(self, game):
        entity = MessageEntity(type=MessageEntity.URL, offset=13, length=17)
        game.text_entities = [entity]

        assert game.parse_text_entity(entity) == "http://google.com"

    def test_parse_entities(self, game):
        entity = MessageEntity(type=MessageEntity.URL, offset=13, length=17)
        entity_2 = MessageEntity(type=MessageEntity.BOLD, offset=13, length=1)
        game.text_entities = [entity_2, entity]

        assert game.parse_text_entities(MessageEntity.URL) == {entity: "http://google.com"}
        assert game.parse_text_entities() == {entity: "http://google.com", entity_2: "h"}
