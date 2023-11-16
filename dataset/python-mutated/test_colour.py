"""
The MIT License (MIT)

Copyright (c) 2015-present Rapptz

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
from __future__ import annotations
import discord
import pytest

@pytest.mark.parametrize(('value', 'expected'), [('0xFF1294', 16716436), ('0xff1294', 16716436), ('0xFFF', 16777215), ('0xfff', 16777215), ('#abcdef', 11259375), ('#ABCDEF', 11259375), ('#ABC', 11189196), ('#abc', 11189196), ('rgb(68,36,59)', 4465723), ('rgb(26.7%, 14.1%, 23.1%)', 4465723), ('rgb(20%, 24%, 56%)', 3358095), ('rgb(20%, 23.9%, 56.1%)', 3358095), ('rgb(51, 61, 143)', 3358095)])
def test_from_str(value, expected):
    if False:
        print('Hello World!')
    assert discord.Colour.from_str(value) == discord.Colour(expected)

@pytest.mark.parametrize('value', ['not valid', '0xYEAH', '#YEAH', '#yeah', 'yellow', 'rgb(-10, -20, -30)', 'rgb(30, -1, 60)', 'invalid(a, b, c)', 'rgb('])
def test_from_str_failures(value):
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError):
        discord.Colour.from_str(value)