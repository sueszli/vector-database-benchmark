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

from telegram import InputContactMessageContent, User
from tests.auxil.slots import mro_slots


@pytest.fixture(scope="module")
def input_contact_message_content():
    return InputContactMessageContent(
        TestInputContactMessageContentBase.phone_number,
        TestInputContactMessageContentBase.first_name,
        TestInputContactMessageContentBase.last_name,
    )


class TestInputContactMessageContentBase:
    phone_number = "phone number"
    first_name = "first name"
    last_name = "last name"


class TestInputContactMessageContentWithoutRequest(TestInputContactMessageContentBase):
    def test_slot_behaviour(self, input_contact_message_content):
        inst = input_contact_message_content
        for attr in inst.__slots__:
            assert getattr(inst, attr, "err") != "err", f"got extra slot '{attr}'"
        assert len(mro_slots(inst)) == len(set(mro_slots(inst))), "duplicate slot"

    def test_expected_values(self, input_contact_message_content):
        assert input_contact_message_content.first_name == self.first_name
        assert input_contact_message_content.phone_number == self.phone_number
        assert input_contact_message_content.last_name == self.last_name

    def test_to_dict(self, input_contact_message_content):
        input_contact_message_content_dict = input_contact_message_content.to_dict()

        assert isinstance(input_contact_message_content_dict, dict)
        assert (
            input_contact_message_content_dict["phone_number"]
            == input_contact_message_content.phone_number
        )
        assert (
            input_contact_message_content_dict["first_name"]
            == input_contact_message_content.first_name
        )
        assert (
            input_contact_message_content_dict["last_name"]
            == input_contact_message_content.last_name
        )

    def test_equality(self):
        a = InputContactMessageContent("phone", "first", last_name="last")
        b = InputContactMessageContent("phone", "first_name", vcard="vcard")
        c = InputContactMessageContent("phone_number", "first", vcard="vcard")
        d = User(123, "first", False)

        assert a == b
        assert hash(a) == hash(b)

        assert a != c
        assert hash(a) != hash(c)

        assert a != d
        assert hash(a) != hash(d)
