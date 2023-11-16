# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# flake8: noqa


def foo():
    return "123.456.789.123"


def bar_format_strings():
    user_controlled = 1
    return f"{user_controlled}:123.456.789.123"


def bar_percent_format():
    user_controlled = 1
    return "%s:123.456.789.123" % (user_controlled,)


def bar_dot_format():
    user_controlled = 1
    return "{}:123.456.789.123".format(user_controlled)


def does_not_match():
    return "123.456"


def multiple_patterns():
    return "<123.456.789.123>"


GOOGLE_API_KEY = "AIzaSyB2qiehH9CMRIuRVJghvnluwA1GvQ3FCe4"


def string_source_top_level():
    params = {"key": GOOGLE_API_KEY}
    return params


def string_source_not_top_level():
    params = {"key": "AIzaSyB2qiehH9CMRIuRVJghvnluwA1GvQ3FCe4"}
    return params


def string_source_top_level_local_overwrite():
    GOOGLE_API_KEY = "safe"
    params = {"key": GOOGLE_API_KEY}
    return params


def string_literal_arguments_source(template: str, x):
    if 1 == 1:
        return template.format("SELECT1", 1)
    elif 1 == 1:
        return template % "SELECT2"
    else:
        return x + "SELECT3"

START, BODY, END = "<html>", "lorem ipsum", "</html>"

def toplevel_simultaneous_assignment():
    return START + BODY + END
