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

from telegram import Bot, InlineQuery, Location, Update, User
from tests.auxil.bot_method_checks import (
    check_defaults_handling,
    check_shortcut_call,
    check_shortcut_signature,
)
from tests.auxil.slots import mro_slots


@pytest.fixture(scope="module")
def inline_query(bot):
    ilq = InlineQuery(
        TestInlineQueryBase.id_,
        TestInlineQueryBase.from_user,
        TestInlineQueryBase.query,
        TestInlineQueryBase.offset,
        location=TestInlineQueryBase.location,
    )
    ilq.set_bot(bot)
    return ilq


class TestInlineQueryBase:
    id_ = 1234
    from_user = User(1, "First name", False)
    query = "query text"
    offset = "offset"
    location = Location(8.8, 53.1)


class TestInlineQueryWithoutRequest(TestInlineQueryBase):
    def test_slot_behaviour(self, inline_query):
        for attr in inline_query.__slots__:
            assert getattr(inline_query, attr, "err") != "err", f"got extra slot '{attr}'"
        assert len(mro_slots(inline_query)) == len(set(mro_slots(inline_query))), "duplicate slot"

    def test_de_json(self, bot):
        json_dict = {
            "id": self.id_,
            "from": self.from_user.to_dict(),
            "query": self.query,
            "offset": self.offset,
            "location": self.location.to_dict(),
        }
        inline_query_json = InlineQuery.de_json(json_dict, bot)
        assert inline_query_json.api_kwargs == {}

        assert inline_query_json.id == self.id_
        assert inline_query_json.from_user == self.from_user
        assert inline_query_json.location == self.location
        assert inline_query_json.query == self.query
        assert inline_query_json.offset == self.offset

    def test_to_dict(self, inline_query):
        inline_query_dict = inline_query.to_dict()

        assert isinstance(inline_query_dict, dict)
        assert inline_query_dict["id"] == inline_query.id
        assert inline_query_dict["from"] == inline_query.from_user.to_dict()
        assert inline_query_dict["location"] == inline_query.location.to_dict()
        assert inline_query_dict["query"] == inline_query.query
        assert inline_query_dict["offset"] == inline_query.offset

    def test_equality(self):
        a = InlineQuery(self.id_, User(1, "", False), "", "")
        b = InlineQuery(self.id_, User(1, "", False), "", "")
        c = InlineQuery(self.id_, User(0, "", False), "", "")
        d = InlineQuery(0, User(1, "", False), "", "")
        e = Update(self.id_)

        assert a == b
        assert hash(a) == hash(b)
        assert a is not b

        assert a == c
        assert hash(a) == hash(c)

        assert a != d
        assert hash(a) != hash(d)

        assert a != e
        assert hash(a) != hash(e)

    async def test_answer_error(self, inline_query):
        with pytest.raises(ValueError, match="mutually exclusive"):
            await inline_query.answer(results=[], auto_pagination=True, current_offset="foobar")

    async def test_answer(self, monkeypatch, inline_query):
        async def make_assertion(*_, **kwargs):
            return kwargs["inline_query_id"] == inline_query.id

        assert check_shortcut_signature(
            InlineQuery.answer, Bot.answer_inline_query, ["inline_query_id"], ["auto_pagination"]
        )
        assert await check_shortcut_call(
            inline_query.answer, inline_query.get_bot(), "answer_inline_query"
        )
        assert await check_defaults_handling(inline_query.answer, inline_query.get_bot())

        monkeypatch.setattr(inline_query.get_bot(), "answer_inline_query", make_assertion)
        assert await inline_query.answer(results=[])

    async def test_answer_auto_pagination(self, monkeypatch, inline_query):
        async def make_assertion(*_, **kwargs):
            inline_query_id_matches = kwargs["inline_query_id"] == inline_query.id
            offset_matches = kwargs.get("current_offset") == inline_query.offset
            return offset_matches and inline_query_id_matches

        monkeypatch.setattr(inline_query.get_bot(), "answer_inline_query", make_assertion)
        assert await inline_query.answer(results=[], auto_pagination=True)
